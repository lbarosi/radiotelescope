# Flask deploy

```bash
#/etc/systemd/system/my-server.service
[Unit]
Description=Flask Web Application Server using Gunicorn
After=network.target

[Service]
User=lbarosi
Group=www-data
WorkingDirectory=/home/lbarosi/BAROSI/HOME/6000_PYTHONIA/2_Doing/radiotelescope
Environment="PATH=/home/lbarosi/BAROSI/HOME/6000_PYTHONIA/2_Doing/radiotelescope/env/bin"
ExecStart=/home/lbarosi/BAROSI/HOME/6000_PYTHONIA/2_Doing/radiotelescope/env/bin/gunicorn -w 3 --bind 127.0.0.1:5000:/tmp/app.sock wsgi:app'
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash

```
