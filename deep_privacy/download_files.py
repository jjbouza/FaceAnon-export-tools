from google_drive_downloader import GoogleDriveDownloader as gdd

# download default_cpu.ckpt and large_cpu.ckpt
file_ids = {"default_cpu.ckpt":   "1nDboo1Z4pgsZ08N5WpN-VbWFxPkOu2Gv",
            "large_cpu.ckpt":     "1MfflN1OubTIx0OtoQQ6rW8-zBxjZArz0", 
            "default_config.yml": "1cCSfgXanNT_hhaeYr2oB76JqfuyHZHcZ"}
            #"large_config.yml": ""}

for key, val in file_ids.items():
    gdd.download_file_from_google_drive(file_id=val,
                                            dest_path='./deep_privacy/'+key, unzip=False)
