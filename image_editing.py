from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import os

client = genai.Client()

prompt_path = 'face_edit_prompts_49_with_category.csv'

with open(prompt_path, 'r') as file:
    prompt = []
    for line in file:
        prompt.append(line.strip().split(",")[1])
        # print(prompt)


input_image_folder = '/media/ee303/disk1/dataset/cemi_id'

for parent, subdirs, files in os.walk(input_image_folder):
    for file in files:
        if file.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(parent, file)
            for i, single_prompt in enumerate(prompt):
                image = Image.open(image_path)
                # print(os.path.splitext(file)[0], i, single_prompt)
                image_id = parent.split("/")[-1]
                print(f"------ Processing the image id: {image_id} ------")
                response = client.models.generate_content(
                    model="gemini-2.5-flash-image-preview",
                    contents=[single_prompt, image],
                )

                for part in response.candidates[0].content.parts:
                    if part.text is not None:
                        print(part.text)
                    elif part.inline_data is not None:
                        os.makedirs(f"gemini_{image_id}", exist_ok=True)
                        out_name = f"gemini_{image_id}/{i}.png"
                        image = Image.open(BytesIO(part.inline_data.data))
                        image.save(out_name)