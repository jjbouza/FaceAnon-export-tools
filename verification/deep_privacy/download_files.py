from google_drive_downloader import GoogleDriveDownloader as gdd

# download default_cpu.ckpt and large_cpu.ckpt
file_ids = {"WIDERFace_DSFD_RES152.pth": "1WeXlNYsM6dMP3xQQELI-4gxhwKUQxc3-"}

for key, val in file_ids.items():
    gdd.download_file_from_google_drive(file_id=val,
                                            dest_path='./test/deep_privacy/detection/dsfd/weights/'+key, unzip=False)
