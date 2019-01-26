# vision-2019

### Setup instructions:

1. Image Raspberry pi with WPIlib Pi Image
2. Plug in cameras in the order as specified below
3. On the Vision Settings page, click add connected camera and choose the by path option in order. If the cameras are plugged in, there should be two logitech cameras first, then two PSeye cameras.
4. Upload camera configs (Or manually configure (info below) and click the copy button, and then save)
5. In 'Application Settings', select 'Uploaded Python File' and upload cameraServer.py to the box
6. Change the Pi to Writable and click save

### Camera Usb Port Order:

[ Top Cargo  ][ Top Hatch  ]
[Ground Cargo][Ground Hatch]

### Camera settings:

The Playstation eye cameras should have all of the auto settings to off, and the exposure to 9. They have an issue which is that the exposure default often doesn't work. We have to programmatically set this multiple times

The Logitech cameras should also have all auto settings to off or manual, and the exposure should be 1