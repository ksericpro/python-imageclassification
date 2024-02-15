import gradio
import gradio as gr

#######################
## Step 1 Load Model
#######################


##########################
## Step 2 Predict
##########################
#create a function to make predictions
#return a dictionary of labels and probabilities
def cat_or_dog(img):
    img = img.reshape(1, 100, 100, 1)
    prediction = model.predict(img).tolist()[0]
    class_names = ["Dog", "Cat"]
    return {class_names[i]: prediction[i] for i in range(2)}
#set the user uploaded image as the input array
#match same shape as the input shape in the model
im = gradio.inputs.Image(shape=(100, 100), image_mode='L', invert_colors=False, source="upload")
#setup the interface
iface = gr.Interface(
    fn = cat_or_dog, 
    inputs = im, 
    outputs = gradio.outputs.Label(),
)
iface.launch(share=True)