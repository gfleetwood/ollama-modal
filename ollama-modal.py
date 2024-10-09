import modal
import os
import subprocess
import time
from typing import Dict
from modal import build, enter, method

#MODEL = os.environ.get("MODEL", "llama3:instruct") 
MODEL = os.environ.get("MODEL", "llava-llama3")

def pull(model: str = MODEL):

    # ollama_path = subprocess.run(["which", "ollama"], capture_output=True, text=True).stdout.strip()
    # print(ollama_path)

    subprocess.run(["systemctl", "daemon-reload"])
    subprocess.run(["systemctl", "enable", "ollama"])
    subprocess.run(["systemctl", "start", "ollama"])
    time.sleep(5)  # 2s, wait for the service to start

    subprocess.run(["ollama", "pull", model], stdout = subprocess.PIPE)

image = (
    modal.Image
    .debian_slim()
    .apt_install("curl", "systemctl")
    .run_commands( # from https://github.com/ollama/ollama/blob/main/docs/linux.md
        'curl -fsSL https://ollama.com/install.sh | sh'
        #"curl -L https://ollama.com/download/ollama-linux-amd64 -o /usr/bin/ollama",
        #"chmod +x /usr/bin/ollama",
        #"useradd -r -s /bin/false -m -d /usr/share/ollama ollama",
    )
    .copy_local_file("ollama.service", "/etc/systemd/system/ollama.service")
    .pip_install("ollama")
    .run_function(pull)
)

app = modal.App(name="ollama", image=image)

with image.imports():
    import ollama

@app.cls(gpu="a10g", container_idle_timeout=300)
class Ollama:
    @build()
    def pull(self):
        # TODO(irfansharif): Was hoping that the following would use an image
        # with this explicit @build() step's results, but alas, it doesn't - so
        # we're baking it directly into the base image above. Also, would be
        # nice to simply specify the class name? Not like the method is
        # specified has any relevance.
        #
        #  $ modal shell ollama-modal.py::Ollama.infer

        # pull(model=MODEL)
        ...

    @enter()
    def load(self):
        subprocess.run(["systemctl", "start", "ollama"])

    @method()
    def infer(self, text: str):
        stream = ollama.chat(
            model=MODEL,
            messages=[{'role': 'user', 'content': text}],
            stream=True
        )
        for chunk in stream:
            yield chunk['message']['content']
            #print(chunk['message']['content'], end='', flush=True)
        return
    
    @method()
    def run_inference(self, base64_image: str):
        stream = ollama.chat(
            model=MODEL,
            messages=[{
                "role": "user", 
                "content": 'transcribe the image in lowercase', 
                #"images": [f"data:image/jpeg;base64,{base64_image}"],
                "images": [base64_image]
            }],
            stream=True,
        )
        
        for chunk in stream:
            yield chunk['message']['content']
            #print(chunk["message"]["content"], end="", flush=True)

# Convenience thing, to run using:
#
#  $ modal run ollama-modal.py [--lookup] [--text "Why is the sky blue?"]
@app.local_entrypoint()
def main(text: str = "Why is the sky blue?", lookup: bool = False):
    if lookup:
        ollama = modal.Cls.lookup("ollama", "Ollama")
    else:
        ollama = Ollama()
    for chunk in ollama.infer.remote_gen(text):
        print(chunk, end='', flush=False)

# @app.function(memory = 4000, cpu = 2.0)
# @modal.web_endpoint(method = "POST")
# def fx(payload: Dict):
#     '''
#     payload = {'search_term': search_input.value.strip()}
#     requests.post(endpoint, json = payload)
#     '''

#     import base64
    
#     x = base64.b64decode(payload['img'])
#     ollama = Ollama()

#     for chunk in ollama.run_inference.remote_gen(x):
#         print(chunk, end='', flush=False)

#     return True