# Set up to import files

from google.colab import auth
auth.authenticate_user()
import gspread
from oauth2client.client import GoogleCredentials

# Import PyDrive and associated libraries.
# This only needs to be done once per notebook.
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# Authenticate and create the PyDrive client.
# This only needs to be done once per notebook.
gauth = GoogleAuth()
drive = GoogleDrive(gauth)
gauth.credentials = GoogleCredentials.get_application_default()
gc = gspread.authorize(gauth.credentials)