import gradio
import gradio as gr
import pickle

#######################
## Step 1 Load Model
#######################
FILENAME = 'models/pets/finalized_model2.sav'

print("Loading model {}".format(FILENAME))
# open a file, where you stored the pickled data
file = open(FILENAME, 'rb')

# dump information to that file
model = pickle.load(file)

# close the file
file.close()


##########################
## Step 2 Predict
##########################

#create a function to make predictions
#return a dictionary of labels and probabilities
def cat_or_dog(img):
    img = img.reshape(1, 150, 150, 1)
    prediction = model.predict(img).tolist()[0]
    class_names = ["Dog", "Cat"]
    return {class_names[i]: prediction[i] for i in range(2)}
#set the user uploaded image as the input array
#match same shape as the input shape in the model
#im = gradio.components.Image(shape=(100, 100), image_mode='L', invert_colors=False, source="upload")
im = gradio.inputs.Image(shape=(150, 150), image_mode='L', invert_colors=False, source="upload")

#setup the interface
iface = gr.Interface(
    #inputs=gr.Image(type="pil"),
    #outputs=gr.Label(num_top_classes=3),
    fn = cat_or_dog, 
    inputs = im, 
    outputs = gradio.outputs.Label(),
)
iface.launch(share=True)