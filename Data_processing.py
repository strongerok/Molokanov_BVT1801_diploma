import os

import dateutil.parser
import geopandas
import pandas as pd

from Photo_processing import help_sort


# function gets info from isoformat (year, month etc.)
def get_info_from_isoformat(isoformat, info):
    d = dateutil.parser.parse(isoformat)
    return d.strftime(info)


# cleaning dataset, splitting it into 2 datasets, for training (2020y) and testing (2021y)
def get_train_test_dataframes_from_dataset():
    pd.set_option('display.max_columns', None)
    pd.options.mode.chained_assignment = None
    accidents = geopandas.read_file("Moscow_car_accidents.geojson")  # file
    accidents['longitude'] = round(accidents.geometry.x, 6).astype(str)
    accidents['latitude'] = round(accidents.geometry.y, 6).astype(str)
    accidents['grid_square'] = accidents[['longitude', 'latitude']].agg(','.join, axis=1)

    print("Dropping unnecessary columns")
    accidents.drop(columns=["id", "region", "scheme", "address", "parent_region", "geometry", "point", "category"],
                   axis=1,
                   inplace=True)
    accidents.rename(columns={'participants_count': 'participants', 'injured_count': 'injured', 'dead_count': 'dead'},
                     inplace=True)
    accidents = pd.DataFrame(accidents)

    print("changing severity to numbers")
    accidents["severity"].replace(["Легкий", "Тяжёлый", "С погибшими"], [0, 1, 2], inplace=True)

    print("changing light to numbers")
    accidents["light"].replace(["Светлое время суток", "В темное время суток, освещение включено",
                                "Сумерки", "В темное время суток, освещение отсутствует",
                                "В темное время суток, освещение не включено"], [0, 1, 2, 3, 4], inplace=True)

    print("creating \"month\" column")
    print("creating \"safe\" column")
    print("creating train and test datasets using 2020 and 2021 car accidents respectively")
    accidents.loc[:, "month"] = ""
    accidents.loc[:, "safe"] = ""
    train_df = pd.DataFrame(columns=list(accidents))
    test_df = pd.DataFrame(columns=list(accidents))
    for count, date in enumerate(accidents["datetime"]):
        if int(get_info_from_isoformat(date, "%Y")) > 2016:
            accidents['safe'][count] = 0
            accidents["month"][count] = get_info_from_isoformat(date, "%m")
            accidents["datetime"][count] = get_info_from_isoformat(date, "%Y")
            if get_info_from_isoformat(date, "%Y") == "2021":
                test_df.loc[test_df.shape[0]] = accidents.iloc[count]
                print(count)
            else:
                train_df.loc[train_df.shape[0]] = accidents.iloc[count]
                print(count)
        else:
            continue

    train_df.drop(columns=["datetime"], axis=1, inplace=True)
    test_df.drop(columns=["datetime"], axis=1, inplace=True)
    train_df = train_df.reindex(columns=['longitude', 'latitude', 'month', 'light', 'severity',
                                         'participants', 'injured', 'dead', 'grid_square', 'safe'])
    test_df = test_df.reindex(columns=['longitude', 'latitude', 'month', 'light', 'severity',
                                       'participants', 'injured', 'dead', 'grid_square', 'safe'])

    print("saving datasets into .csv files")
    train_df.to_csv("train_2020.csv", index=False)
    test_df.to_csv("test_2021.csv", index=False)
    print("done!")


# getting minimal and maximal longitudes and latitudes of training dataset points
def get_min_max_grid_squares(name):
    df = pd.read_csv(name)
    return [f'{df["longitude"].min()},{df["latitude"].min()}', f'{df["longitude"].max()}, {df["latitude"].max()}']


def sort_dataset_by_images(name, list_of_paths):
    dataset = pd.read_csv(name)
    images = help_sort(list_of_paths[0])
    if len(list_of_paths) > 1:
        for i in range(1, len(list_of_paths)):
            next_images = help_sort(list_of_paths[i])
            images = pd.concat([images, next_images], ignore_index=True)
    final_dataset = pd.DataFrame(columns=list(dataset))
    print(images.head(10))
    print("started sorting...")
    for img in range(0, images.shape[0]):
        size = dataset.shape[0]
        for data in range(0, size):
            if (images["min_lon"][img] <= dataset["longitude"][data] <= images["max_lon"][img]) & (
                    images["min_lat"][img] <= dataset["latitude"][data] <= images["max_lat"][img]):
                final_dataset.loc[final_dataset.shape[0]] = dataset.iloc[data]
                dataset = dataset.drop(index=dataset.index[data], axis=0)
                dataset = dataset.reset_index(drop=True)
                break
    print("saved!")
    final_dataset.to_csv(name, index=False)


# getting full dataset with good areas, even tho they don`t give any info to model
def connect_good_to_bad_areas_dataset(bad_dataset, good_path):
    name = bad_dataset[:-4] + "_bad_good_merged.csv"
    bad = pd.read_csv(bad_dataset)
    bad.drop(columns=['Unnamed: 0'], axis=1, inplace=True)
    good = pd.DataFrame(columns=list(bad))
    for filename in os.listdir(good_path):
        if filename.endswith(".jpg"):
            array = [filename[:-4].split(",")[0], filename[:-4].split(",")[1], 0, -1, -1, 0, 0, 0, filename[:-4], 1]
            good.loc[good.shape[0]] = array
    all_squares_accidents = pd.concat([bad, good], ignore_index=True)

    all_squares_accidents.to_csv(name, index=False)
    return all_squares_accidents.shape[0]
