# Prompt Generation using GPT-3 + Stable Diffusion

 This example showcases how to use the component to enhance the prompts in the Stable Diffusion (SD), using a simple user interface that allows you to interact with the model and see the enhanced prompt and generates images. 
 

 ## Step 1:
If you don't have Lightning installed yet, go ahead and install it using:

``` bash

pip install -U lightning

```

## Step 2

Once Lightning is installed, clone the repo and cd into examples/stable-diffusion


``` bash

git clone https://github.com/Lightning-AI/lightning-gpt3
cd lightning-gpt3/examples/stable-diffusion

```
## Step 3:


**_NOTE:_**  This process will download the stable diffusion weights from s3, which are 4.27 GB in size. The time it takes will depend on your internet speed 

To run the  app locally using the command:

 ``` bash
 lightning run app app.py --setup
 ```

If you prefer to run it in the cloud use:


 ``` bash
 lightning run app app.py --setup --cloud
 ```

Once the model is ready, your user interface will open in your web browser. This will allow you to interact with the model and see its outputs.
