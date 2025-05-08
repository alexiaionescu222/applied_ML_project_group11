# predict.py
import requests

API_URL = "http://127.0.0.1:8000/predict"
WAV_PATH = "audio_split/test/pop/pop.00088.wav"  # adjust as needed

def main():
    with open(WAV_PATH, "rb") as f:
        files = {"file": f}
        resp  = requests.post(API_URL, files=files)
    if resp.status_code == 200:
        data = resp.json()
        print(f"Predicted genre: {data['genre']}")
    else:
        print("Error:", resp.status_code, resp.text)

if __name__ == "__main__":
    main()