# Download the helper library from https://www.twilio.com/docs/python/install
import os
from twilio.rest import Client
from dotenv import load_dotenv
load_dotenv(override=True)
# Find your Account SID and Auth Token at twilio.com/console
# and set the environment variables. See http://twil.io/secure

account_sid=os.getenv("TWILIO_ACCOUNT_SID")
auth_token=os.getenv("TWILIO_AUTH_TOKEN")
client = Client(account_sid, auth_token)
try:
    call = client.calls.create(
            twiml=open("streams.xml").read(),
        to=os.getenv("MY_PHONE_NUMBER"),
        from_=os.getenv("TWILIO_PHONE_NUMBER"),
    )
    print(call.sid)
except Exception as e:
    print(f"Error making call: {e}")
