import gzip
import urllib.request

import nibabel as nib

def load_nii_gz_from_url(url: str):
    with urllib.request.urlopen(url) as response:
        with gzip.GzipFile(fileobj=response) as uncompressed:
            file_content = uncompressed.read()
            img = nib.Nifti1Image.from_bytes(file_content)
    return img
