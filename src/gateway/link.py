from linkkit import linkkit
import os

lk = linkkit.LinkKit(host_name=os.environ.get('host_name'),
                product_key=os.environ.get('product_key'),
                device_name=os.environ.get('device_name'),
                device_secret=os.environ.get('device_secret'))



def on_connect(session_flag, rc, userdata):
    print("on_connect:%d,rc:%d,userdata:" % (session_flag, rc))
    pass

def on_disconnect(rc, userdata):
    print("on_disconnect:rc:%d,userdata:" % rc)

lk.on_connect = on_connect
lk.on_disconnect = on_disconnect

lk.connect_async()



while True:
    pass
