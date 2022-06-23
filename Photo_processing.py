import shutil
import sys
import os
import requests
import math
import pip
from keras.preprocessing.image import img_to_array, load_img
from matplotlib import pyplot as plt

try:
    import numpy as np
    import pandas as pd
except ImportError:
    pip.main(['install', 'numpy'])
    pip.main(['install', 'pandas'])
    import numpy as np
    import pandas as pd
try:
    from geopandas.tools import geocode, reverse_geocode
    from shapely.geometry import Point
except ModuleNotFoundError:
    pip.main(['install', 'geopandas'])
    pip.main(['install', 'shapely'])
    from geopandas.tools import geocode, reverse_geocode
    from shapely.geometry import Point


# address to coordinates
def geocoding(loc):
    location = geocode(loc, provider="nominatim", user_agent='my_request')
    point = location.iloc[0]['geometry']
    try:
        grid_square = f'{round(point.x, 6)},{round(point.y, 6)}'
    except IndexError:
        grid_square = "Can`t find address"
    return grid_square


# coordinates to address
def reverse_geocoding(coords):
    coords = coords.split(",")
    longitude = float(coords[0])
    latitude = float(coords[1])
    df = reverse_geocode(Point(longitude, latitude))
    return df['address'][0]


# 92 is width of image (in m) which is latitude and 87220 is converter from latitude measure to meters
def get_list_of_lats(min_lat, max_lat):
    counter = 1
    delta_lat = 67.3 / 87220  # increment in latitude measure
    kol = math.ceil(abs(max_lat - min_lat) / delta_lat)  # distance from max to min latitude in meters divided by
    # width of image and rounded up to next integer because amount of grid squares can be integer only
    list_of_lats = np.empty(kol + 1)
    list_of_lats[0] = min_lat

    if min_lat < max_lat:  # if going up (min < max) then we should increment delta
        while min_lat < max_lat:
            min_lat += delta_lat
            list_of_lats[counter] = min_lat
            counter += 1
    elif min_lat > max_lat:  # if going down (min > max) then we should decrement delta
        while min_lat > max_lat:
            min_lat -= delta_lat
            list_of_lats[counter] = min_lat
            counter += 1
    for x in range(0, len(list_of_lats)):  # deleting 0 elements
        if list_of_lats[x] == 0.000000:
            list_of_lats = np.delete(list_of_lats, x)
    return list_of_lats


# same here with longitude
def get_list_of_lons(min_lon, max_lon):
    counter = 1
    delta_lon = 110.6 / 80135
    kol = math.ceil(abs(max_lon - min_lon) / delta_lon)  # longitude also depends on angle of it
    list_of_lons = np.empty(kol + 1)
    list_of_lons[0] = min_lon

    if min_lon > max_lon:
        while min_lon > max_lon:
            min_lon -= delta_lon
            list_of_lons[counter] = min_lon
            counter += 1
    elif min_lon < max_lon:
        while min_lon < max_lon:
            min_lon += delta_lon
            list_of_lons[counter] = min_lon
            counter += 1
    for x in range(0, len(list_of_lons)):
        if list_of_lons[x] == 0.000000:
            list_of_lons = np.delete(list_of_lons, x)
    return list_of_lons


# getting all points from min to max lon and lat
def get_dataframe_of_coords(grid_square_start="", grid_square_end=""):
    min_longitude, min_latitude = 37.710735, 55.755946
    max_longitude, max_latitude = 37.717848, 55.750821

    # if user inputted addresses
    if grid_square_start != "":
        start_coords = grid_square_start.split(",")
        min_longitude, min_latitude = float(start_coords[0]), float(start_coords[1])
        end_coords = grid_square_end.split(",")
        max_longitude, max_latitude = float(end_coords[0]), float(end_coords[1])

    # Getting a list of all grid squares (258x118.6m)
    lats = get_list_of_lats(min_latitude, max_latitude)
    lons = get_list_of_lons(min_longitude, max_longitude)
    coords = [(round(x, 6), round(y, 6)) for x in lats for y in lons]

    # Converting to a dataframe and adding a column for the grid square name
    coords = pd.DataFrame(coords, columns=['latitude', 'longitude'])
    coords['grid_square'] = coords['longitude'].map(str) + "," + coords['latitude'].map(str)
    return coords


# using coordinates of all points to download fragments of satellite map (center in point(lon, lat))
def load_images(coordinates):
    for i in range(0, coordinates['grid_square'].size):  # for each grid square
        map_request = "http://static-maps.yandex.ru/1.x/?ll={ll}&z={z}&l={type}&size=256,256".format(
            ll=coordinates.iloc[i]['grid_square'], z=18,
            type="map")  # getting yandex map centered in grid square coordinates
        x = coordinates.iloc[i]['grid_square']
        response = requests.get(map_request)
        if not response:  # errors handler
            print("Request execution error:")
            print(map_request)
            print("Http status:", response.status_code, "(", response.reason, ")")
            sys.exit(1)

        # Writing image into the file
        myloc = r"C:\Users\Stronger\Desktop\Diploma\try1\images\source"
        map_file = f'{x}.jpg'
        full_path = os.path.join(myloc, map_file)
        try:
            with open(full_path, "wb") as file:
                file.write(response.content)
        except IOError as ex:
            print("Error writing temporary file:", ex)
            sys.exit(2)


def show_image(image_path, size=None):
    if size is None:
        size = [600, 400]
    img = load_img(image_path, target_size=(size[0], size[1]))
    img_tensor = img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    plt.imshow(img_tensor[0])


def show_images(images_path):
    counter = 0
    files = os.listdir(images_path)
    images = [x for x in files if x.endswith('.jpg')]
    fig = plt.figure(figsize=(16, 7))
    for i in range(1, len(images) + 1):
        file = str(i) + ".jpg"
        file_path = os.path.join(images_path, file)
        if os.path.exists(file_path):
            fig.add_subplot(1, 5, counter + 1)
            plt.suptitle('Example areas where serious accidents occurred', fontsize=15)
            show_image(file_path)
            plt.axis('off')
            counter += 1
    plt.show()


# function for connecting right way and connecting addresses
def connect_address(city, street, home_number):
    # deleting common words
    city = city.lower().replace("город ", "")
    city = city.replace(" город", "")
    street = street.replace("улица ", "")
    street = street.lower().replace(" улица", "")
    home_number = home_number.lower().replace(" ", "")
    home_number = home_number.replace("дом", "")
    home_number = home_number.replace("строение", "с")
    home_number = home_number.replace("корпус", "к")
    # correcting "к" and "с" because there should be a space before them
    if home_number.lower().find("к") != -1:
        chars = list(home_number)
        for position, symbol in enumerate(home_number):
            try:
                float(symbol)
                position += 1
            except ValueError:
                chars.insert(position, " ")
                position += 2
        home_number = "".join(chars)
    elif home_number.lower().find("с") != -1:
        chars = list(home_number)
        for position, symbol in enumerate(home_number):
            try:
                float(symbol)
                position += 1
            except ValueError:
                chars.insert(position, " ")
                position += 2
        home_number = "".join(chars)
    return f'{city} {street} {home_number}'


# function gets info from user (starting and ending addresses) and gives images of satellite map
def get_images(list_of_grid=None):
    loc_city = input("Input city of start point or \"no\" if don`t want to:\n")
    if str.lower(loc_city) == "no":
        if list_of_grid is not None:
            coordinates_start, coordinates_end = list_of_grid
            print(get_dataframe_of_coords(coordinates_start, coordinates_end))
            load_images(get_dataframe_of_coords(coordinates_start, coordinates_end))
            coords = [[coordinates_start[0], coordinates_end[0]], [coordinates_start[1], coordinates_end[1]]]
        else:
            print("if you don`t want to enter address, give me points then")
            return
    else:
        loc_street = input("Input street of start point:\n")
        loc_home_number = input("Input home number of start point:\n")
        coordinates_start = geocoding(connect_address(loc_city, loc_street, loc_home_number))
        print(coordinates_start)
        loc_city = input("Input city of end point:\n")
        loc_street = input("Input street of end point:\n")
        loc_home_number = input("Input home_number of end point:\n")
        coordinates_end = geocoding(connect_address(loc_city, loc_street, loc_home_number))
        print(coordinates_end)
        if (coordinates_start != "Can`t find address") & (coordinates_end != "Can`t find address"):
            print(get_dataframe_of_coords(coordinates_start, coordinates_end))
            load_images(get_dataframe_of_coords(coordinates_start, coordinates_end))
            print("\nSuccess!")
            coords = [[coordinates_start[0], coordinates_end[0]], [coordinates_start[1], coordinates_end[1]]]
        else:
            print("\nCan`t find one of addresses")
            return
    return coords


# creates pandas dataframe with min max lons and lats, grid square and name of each image
# used to create dataset which helps sort images by checking if coords of accident belongs to image
def help_sort(path="images/source/"):
    delta_lat = 67.3 / 87220 / 2
    delta_lon = 110.6 / 80135 / 2
    image_edges = pd.DataFrame(columns=["min_lon", "max_lon", "min_lat", "max_lat", "image_name"])
    for filename in os.listdir(path):
        if filename.endswith(".jpg"):
            lon, lat = filename[:-4].split(",")
            array = [(round(float(lon), 6) - delta_lon), (round(float(lon), 6) + delta_lon),
                     (round(float(lat), 6) - delta_lat), (round(float(lat), 6) + delta_lat), filename]
            image_edges.loc[image_edges.shape[0]] = array
    return image_edges


# function sorts images in "bad" folder if at least 1 point from incidents dataset lays on it or "good" folder if not
def sort_images_with_dataset(coords, file_name: str, bad_dest_path: str, good_dest_path: str, help_path):
    print("sorting images for training")
    path = r"images/all_images/all/"
    source = r"images/source/"
    dataset = pd.read_csv(file_name)
    final = pd.DataFrame(columns=list(dataset))
    images = help_sort(help_path)
    if os.path.isdir(path):
        if not os.listdir(path):
            for filename in os.listdir(source):
                if filename.endswith(".jpg"):  # copying all images in "all" folder for sorting
                    shutil.copy2(source + filename, path)
    for num in range(0, dataset["safe"].size):
        if (dataset["longitude"][num] > min(coords[0])) & (
                dataset["longitude"][num] < max(coords[0])) & (dataset["latitude"][num] > min(coords[1])) & (
                dataset["latitude"][num] < max(coords[1])):
            # for each accident point check what image it belongs to and copy it to "bad" folder
            for count, val in enumerate(images["image_name"]):
                if (images["min_lon"][count] <= dataset["longitude"][num] <= images["max_lon"][count]) & (
                        images["min_lat"][count] <= dataset["latitude"][num] <= images["max_lat"][count]):
                    if os.path.exists(path + images["image_name"][count]):
                        shutil.copy2(path + images["image_name"][count], bad_dest_path)
                        os.remove(path + images["image_name"][count])
                        final.loc[final.shape[0]] = dataset.iloc[num]
                        break
    for filename in os.listdir(path):
        if filename.endswith(".jpg"):  # all other images are "good" so copy them into "good" folder
            shutil.copy2(path + filename, good_dest_path)
            os.remove(path + filename)
    file_name = file_name[:-4] + "_after_images_sort.csv"
    final.to_csv(file_name, index=False)  # save dataset into .csv file
    return dataset
