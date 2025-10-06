from huggingface_hub import HfFolder
import os
# Set the token if provided
HfFolder.save_token(os.environ['HF_API_KEY'])