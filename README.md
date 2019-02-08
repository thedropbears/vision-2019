# vision-2019

### Setup instructions:

1. Image Raspberry pi with WPIlib Pi Image
2. Plug in cameras in the order as specified below
3. On the Vision Settings page, click add connected camera and choose the by path option in order. If the cameras are plugged in, the order should be Cargo (Logitech C920) then Hatch (Logitech C920) (Top Two Ports)
4. Upload camera configs (Or manually configure (info below) and click the copy button, and then save) (Note that all cameras must have unique names)
5. In 'Application Settings', select 'Uploaded Python File' and upload cameraServer.py to the box
6. Change the Pi to Writable and click save

### Camera settings:

The Logitech cameras should have all auto settings to off or manual, and the exposure should be 1