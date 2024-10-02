import requests
import os
import tarfile


def download_vrplib():
    sets = ["A", "B", "E", "F", "M", "P", "X"]
    URL = "http://vrp.galgos.inf.puc-rio.br/media/com_vrp/instances/Vrp-Set-{}.tgz"

    for s in sets:
        url = URL.format(s)
        print(f"Downloading {url}")

        # Download the file
        r = requests.get(url, stream=True)

        # Check if the request was successful
        if r.status_code == 200:
            # Check the content type
            content_type = r.headers.get('Content-Type')
            if 'gzip' in content_type:
                # Save the file
                with open(f"Vrp-Set-{s}.tgz", "wb") as f:
                    f.write(r.content)

                # Extract the file under the vrplib folder
                with tarfile.open(f"Vrp-Set-{s}.tgz", "r") as tar:
                    tar.extractall(path="vrplib")

                # Remove the downloaded file
                os.remove(f"Vrp-Set-{s}.tgz")
            else:
                print(f"Warning: File {url} is not a gzip format. Skipping extraction.")
        else:
            print(f"Error: Failed to download {url}. Status code: {r.status_code}")

if __name__ == "__main__":
    download_vrplib()