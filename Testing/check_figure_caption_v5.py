# the same as check figures however using the new databse to see if the issue is fixed.

import json
import pymupdf


image_file = "DOC082.pdf-0145-00.png"




document = image_file.split(".pdf")[0]
page_number = int(image_file.split("-")[1])

captions = json.load(open("/Users/woolf/Desktop/dissertation/build_v5/figure_captions_cache.json"))


print("")
print("PICTURE:", image_file)
print("")
print("The picture itself is saved at:")
print("build_v5/extracted_figures/" + document + "/" + image_file)

print("")
print("WHAT THE AI WROTE ABOUT IT:")
print("")

if image_file in captions:
    print(captions[image_file])
else:
    print("This picture was not described. It was filtered out before captioning.")


pdf = pymupdf.open("/Users/woolf/Desktop/dissertation/corpus/pdfs/" + document + ".pdf")
page = pdf[page_number]

print("")
print("WHAT IS ACTUALLY ON PAGE", page_number + 1, "OF THE DOCUMENT:")
print("")
print(" ".join(page.get_text().split())[:450])

print("")
print("Photographs embedded on that page:", len(page.get_images()))
print("")
