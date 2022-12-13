# Prompt Generation using GPT-3 + Stable Diffusion

 This example showcases how to use the component to enhance the prompts in the Stable Diffusion (SD). Using these prompts, Stable Diffusion can generate more detailed and realistic images with the SD algorithm. 

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
To run the  app locally using the command:

### Server:
``` bash
lightning run app app.py --setup
```

### Simple UI:
``` bash
lightning run app app_simple_UI.py --setup
```

**_NOTE:_**  This process will download the stable diffusion weights from s3, which are 4.27 GB in size. The time it takes will depend on your internet speed

If you prefer to run it in the cloud use:

### Server:
``` bash
lightning run app app.py --setup --cloud
```
### Simple UI:
``` bash
lightning run app app_simple_UI.py --setup --cloud
```

## Step 4
Once your server is ready, check the Lightning tap (locally) or open your app (cloud). Then, use the predict API address to make requests to your server. To do this, you can use the code snippet provided in the tab, or use the client.py file in the examples/stable-diffusion directory. Just remember to update the URL to your current predict URL. 

Then, run:
```  bash

python client.py

```
