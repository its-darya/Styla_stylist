import urllib.request
import os
from gradio_client import Client, handle_file

# Download a clean, flat-lay t-shirt image
garm_url = "https://media.istockphoto.com/id/482948743/photo/blank-white-t-shirt-front-with-clipping-path.jpg?s=612x612&w=0&k=20&c=BqOnA7aJteT2kFh3X9G20p9x7dM31Z-gG0sVpLh_o8w="
person_url = "https://raw.githubusercontent.com/yisol/IDM-VTON/main/example/person/00008_00.jpg"

try:
    print("Downloading clean flat-lay T-Shirt...")
    urllib.request.urlretrieve(garm_url, "clean_tshirt.jpg")
    print("Downloading person photo...")
    urllib.request.urlretrieve(person_url, "test_person.jpg")
except Exception as e:
    print("Failed to download images:", e)
    exit(1)

print("Sending to yisol/IDM-VTON...")
try:
    client = Client("yisol/IDM-VTON")
    result = client.predict(
        dict={
            "background": handle_file("test_person.jpg"),
            "layers": [],
            "composite": None
        },
        garm_img=handle_file("clean_tshirt.jpg"),
        garment_des="A stylish white t-shirt",
        is_checked=True,
        is_checked_crop=True,
        denoise_steps=30,
        seed=42,
        api_name="/tryon"
    )
    print("SUCCESS! Result saved at:", result[0] if isinstance(result, (list, tuple)) else result)
except Exception as e:
    print("API FAILED:", e)
