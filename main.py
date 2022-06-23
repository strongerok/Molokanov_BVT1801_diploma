from Photo_processing import *
from Neural_networks import *
from Data_processing import *
import pandas as pd
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    # get_train_test_dataframes_from_dataset()
    # start = input("give me coordinates of starting point in shape: longitude, latitude; or \"no\" if you don`t want to")
    # if start == "no":
    #     coords = get_images()
    # else:
    #     end = input("give me coordinates of ending point in shape: longitude,latitude")
    #     grid = [[start.split(",")[0], start.split(",")[1]], [end.split(",")[0], end.split(",")[1]]]
    #     coords = get_images(grid)
    #
    # sort_images_with_dataset(coords, "train_2020.csv", "images/train/bad_areas", "images/train/good_areas")
    # sort_images_with_dataset(coords, "test_2021.csv", "images/test/bad_areas", "images/test/good_areas")
    #
    # connect_good_to_bad_areas_dataset("train_2020_after_images_sort.csv", "images/train/good_areas")
    # connect_good_to_bad_areas_dataset("test_2021_after_images_sort.csv", "images/test/good_areas")
    #
    # sort_dataset_by_images("train_2020_after_images_sort_bad_good_merged.csv",
    #                        ["images/train/bad_areas", "images/train/good_areas"])
    # sort_dataset_by_images("test_2021_after_images_sort_bad_good_merged.csv",
    #                        ["images/source"])
    #
    # build_model()
    test_model("test_2021_after_images_sort_bad_good_merged.csv")
