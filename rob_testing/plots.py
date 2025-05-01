import os
import pandas as pd
import random
import sys
import matplotlib.pyplot as plt


class Dataset:
    def __init__(self, name: str):
        current_dir = os.path.dirname(__file__)
        data_path = os.path.join(current_dir, "data", name)
        self.path = data_path
        self.data = self._load_dataset()

    def _load_dataset(self) -> None:
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Dataset not found at {self.path}")
        return pd.read_csv(self.path)
    
    def print_columns(self) -> None:
        if not hasattr(self, 'data'):
            raise AttributeError("Data not loaded. Call load() method first.")
        print(self.data.columns)

    def get_data(self, column_name):
        if not hasattr(self, 'data'):
            raise AttributeError("Data not loaded. Call load() method first.")
        if column_name not in self.data.columns:
            raise ValueError(f"Column '{column_name}' not found in dataset.")
        return self.data[column_name]
    
    def print_amount_of_rows(self) -> None:
        if not hasattr(self, 'data'):
            raise AttributeError("Data not loaded. Call load() method first.")
        print(f"Number of rows: {len(self.data)}")

    def get_amount_of_rows(self) -> int:
        if not hasattr(self, 'data'):
            raise AttributeError("Data not loaded. Call load() method first.")
        return len(self.data)
    
    def get_data_by_id(self, id: int) -> pd.Series:
        if not hasattr(self, 'data'):
            raise AttributeError("Data not loaded. Call load() method first.")
        if id < 1 or id > len(self.data):
            raise IndexError(f"ID {id} is out of bounds for dataset with {len(self.data)} rows.")
        return self.data.iloc[id - 1]
    
    def get_row(self, column_name: str, value) -> pd.DataFrame:
        if not hasattr(self, 'data'):
            raise AttributeError("Data not loaded. Call load() method first.")
        if column_name not in self.data.columns:
            raise ValueError(f"Column '{column_name}' not found in dataset.")
        return self.data[self.data[column_name] == value]

folder_path = os.path.join(os.path.dirname(__file__), "data")
dataset_files = os.listdir(folder_path)
dataset_dict = {}
for file in dataset_files:
    dataset_name = file.split(".")[0]
    dataset_dict[dataset_name] = Dataset(file)

races = dataset_dict['races']
laptimes = dataset_dict['lap_times']


def main_menu():
    print("Select an option:")
    print("1. Plot driver positions in a random race")
    print("2. ")
    print("3. Exit")
    choice = input("Enter your choice: ")
    return choice

while True:
    user_choice = main_menu()
    if user_choice == "1":
        amount_of_races = races.get_amount_of_rows()
        random_race_id = random.randint(1, amount_of_races)
        random_race = races.get_row("raceId", random_race_id)
        title = random_race['name'].values[0]
        date = random_race['date'].values[0]
        laptimes_race = laptimes.get_row("raceId", random_race_id)
        amount_of_laps = len(laptimes_race['lap'].unique())
        if amount_of_laps == 0:
            print(ValueError("No laps found")) 
            sys.exit(1)

        drivers = laptimes_race['driverId'].unique()
        plt.figure(figsize=(12, 8))
        drivers_set = dataset_dict['drivers']
        for driver in drivers:
            driver_name = drivers_set.get_row("driverId", driver)['surname'].values[0]
            driver_laps = laptimes_race[laptimes_race['driverId'] == driver]
            plt.plot(driver_laps['lap'], driver_laps['position'], label=driver_name)

        plt.gca().invert_yaxis()  # Invert y-axis to show 1st position at the top
        plt.title(f"Driver Positions Over Time for Race: {title} ({date})")
        plt.xlabel("Lap")
        plt.ylabel("Position")
        plt.legend(title="Drivers", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
    elif user_choice == "2":
        pass

    elif user_choice == "3":
        print("Exiting program.")
        break
    else:
        print("Invalid choice. Please try again.")

