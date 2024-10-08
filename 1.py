from roboflow import Roboflow
rf = Roboflow(api_key="EVLoCzmvGSW10Blbwpn5")
project = rf.workspace().project("hard-hat-sample-ckh0q")
model = project.version(3).model

# infer on a local image
print(model.predict("Screenshot 2023-10-16 143415.png", confidence=40, overlap=30).json())

# visualize your prediction
model.predict("Screenshot 2023-10-16 143415.png", confidence=40, overlap=30).save("prediction1.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())