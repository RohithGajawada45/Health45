#!/bin/bash

# Install the necessary libraries
pip3 install PyMuPDF
pip3 install nltk
pip3 install numpy
pip3 install langchain
pip3 install certifi

# Export the SSL certificate file path for certifi
export SSL_CERT_FILE=$(python3 -m certifi)

# Run a Python script to download the necessary nltk data
python3 -c "
import nltk
import ssl
import certifi

ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')
"

echo "All packages installed successfully!"
