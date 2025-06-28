import numpy as np
import mlflow
import pandas as pd

mlflow.set_tracking_uri("https://dagshub.com/michelpf/fiap-ds-mlops-10dtsr-laptop-pricing-brl.mlflow")

model_uri = "models:/laptop-pricing-model-brl/latest"
model = mlflow.pyfunc.load_model(model_uri)

def prepare_data(data):

    data_processed = []

    data_processed.append(int(data["ram_gb"]))
    data_processed.append(int(data["ssd"]))
    data_processed.append(int(data["hdd"]))
    data_processed.append(int(data["graphic_card"]))
    data_processed.append(int(data["warranty"]))

    data_processed.append(1) if data["brand"] == "asus" else data_processed.append(0)
    data_processed.append(1) if data["brand"] == "dell" else data_processed.append(0)
    data_processed.append(1) if data["brand"] == "hp" else data_processed.append(0)
    data_processed.append(1) if data["brand"] == "lenovo" else data_processed.append(0)
    data_processed.append(1) if data["brand"] == "other" else data_processed.append(0)

    data_processed.append(1) if data["processor_brand"] == "amd" else data_processed.append(0)
    data_processed.append(1) if data["processor_brand"] == "intel" else data_processed.append(0)
    data_processed.append(1) if data["processor_brand"] == "m1" else data_processed.append(0)
    
    data_processed.append(1) if data["processor_name"] == "core i3" else data_processed.append(0)
    data_processed.append(1) if data["processor_name"] == "core i5" else data_processed.append(0)
    data_processed.append(1) if data["processor_name"] == "core i7" else data_processed.append(0)
    data_processed.append(1) if data["processor_name"] == "other" else data_processed.append(0)
    data_processed.append(1) if data["processor_name"] == "ryzen 5" else data_processed.append(0)
    data_processed.append(1) if data["processor_name"] == "ryzen 7" else data_processed.append(0)
    
    data_processed.append(1) if data["os"] == "other" else data_processed.append(0)
    data_processed.append(1) if data["os"] == "windows" else data_processed.append(0)
    
    data_processed.append(1) if data["weight"] == "casual" else data_processed.append(0)
    data_processed.append(1) if data["weight"] == "gaming" else data_processed.append(0)
    data_processed.append(1) if data["weight"] == "thinnlight" else data_processed.append(0)

    data_processed.append(1) if data["touchscreen"] == "0" else data_processed.append(0)
    data_processed.append(1) if data["touchscreen"] == "1" else data_processed.append(0)
   
    data_processed.append(1) if data["ram_type"] == "ddr4" else data_processed.append(0)
    data_processed.append(1) if data["ram_type"] == "other" else data_processed.append(0)

    data_processed.append(1) if data["os_bit"] == "32" else data_processed.append(0)
    data_processed.append(1) if data["os_bit"] == "64" else data_processed.append(0)

    len(data_processed)

    return data_processed

def range_golden_data(prediction, expected, tolerance=0.2):
    lower = expected * (1 - tolerance)
    upper = expected * (1 + tolerance)
    return lower < prediction < upper


def test_golden_data():

    payload = {
        "brand": "dell",
        "processor_brand": "intel",
        "processor_name": "core i5",
        "os": "windows",
        "weight": "casual",
        "warranty": "2",
        "touchscreen": "0",
        "ram_gb": "16",
        "hdd": "0",
        "ssd": "256",
        "graphic_card": "8",
        "ram_type": "ddr4",
        "os_bit": "64"
    }

    columns = [
        'ram_gb', 'ssd', 'hdd', 'graphic_card_gb', 'warranty',
        'brand_asus', 'brand_dell', 'brand_hp', 'brand_lenovo', 'brand_other',
        'processor_brand_amd', 'processor_brand_intel', 'processor_brand_m1',
        'processor_name_core i3', 'processor_name_core i5', 'processor_name_core i7',
        'processor_name_other', 'processor_name_ryzen 5', 'processor_name_ryzen 7',
        'ram_type_ddr4', 'ram_type_other', 'os_other', 'os_windows',
        'os_bit_32-bit', 'os_bit_64-bit',
        'weight_casual', 'weight_gaming', 'weight_thinnlight',
        'touchscreen_0', 'touchscreen_1'
    ]

   
    data_processed = prepare_data(payload)
    data_processed = np.array([data_processed], dtype=np.int64)


    df_input = pd.DataFrame(data_processed, columns=columns)
    
    result = model.predict(df_input)

    result = int(result[0])

    print(f"Prediction: {result}")

    assert range_golden_data(result, 9200), "ensuring golden data range for prediction"

def test_model_load_call():

    payload = {
        "brand": "dell",
        "processor_brand": "intel",
        "processor_name": "core i5",
        "os": "windows",
        "weight": "casual",
        "warranty": "2",
        "touchscreen": "0",
        "ram_gb": "16",
        "hdd": "0",
        "ssd": "256",
        "graphic_card": "8",
        "ram_type": "ddr4",
        "os_bit": "64"
    }

    data_processed = prepare_data(payload)
    data_processed = np.array([data_processed], dtype=np.int64)

    columns = [
        'ram_gb', 'ssd', 'hdd', 'graphic_card_gb', 'warranty',
        'brand_asus', 'brand_dell', 'brand_hp', 'brand_lenovo', 'brand_other',
        'processor_brand_amd', 'processor_brand_intel', 'processor_brand_m1',
        'processor_name_core i3', 'processor_name_core i5', 'processor_name_core i7',
        'processor_name_other', 'processor_name_ryzen 5', 'processor_name_ryzen 7',
        'ram_type_ddr4', 'ram_type_other', 'os_other', 'os_windows',
        'os_bit_32-bit', 'os_bit_64-bit',
        'weight_casual', 'weight_gaming', 'weight_thinnlight',
        'touchscreen_0', 'touchscreen_1'
    ]

    df_input = pd.DataFrame(data_processed, columns=columns)

    result = model.predict(df_input)

    result = int(result[0])

    assert isinstance(result, int), "ensuring model prediction returns an integer"
    assert result > 0, "ensuring model prediction is greater than zero"

test_golden_data()