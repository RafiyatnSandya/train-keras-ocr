import streamlit as st
import keras_ocr
import string
import json
import matplotlib.pyplot as plt

# Load the recognizer with the custom weights
recognizer = keras_ocr.recognition.Recognizer(alphabet=string.printable, weights=None)
recognizer.model.load_weights(Bangkit_CRNN_Model.h5')
pipeline = keras_ocr.pipeline.Pipeline(recognizer=recognizer)

# Function to filter OCR entities
def filter_entities(entities):
    filtered_entities = []
    for entity in entities:
        x1 = round(entity["coordinates"][0][0], 2)
        y1 = round(entity["coordinates"][0][1], 2)
        x2 = round(entity["coordinates"][1][0], 2)
        y2 = round(entity["coordinates"][1][1], 2)
        x3 = round(entity["coordinates"][2][0], 2)
        y3 = round(entity["coordinates"][2][1], 2)
        x4 = round(entity["coordinates"][3][0], 2)
        y4 = round(entity["coordinates"][3][1], 2)
        
        if (x1 > 710.00): 
            continue    
        elif (y1 > 530.00) and len(entity["text"]) <= 3: 
            continue
        elif (x1 > 540.00 and y1 > 300.00) and len(entity["text"]) <= 3: 
            continue
        elif (x1 > 180 and y1 > 95.00 and y4 < 150) and len(entity["text"]) <= 3: 
            continue
        else:
            rounded_coordinates = []
            for coord in entity["coordinates"]:
                rounded_coord = [int(round(coord[0])), int(round(coord[1]))]
                rounded_coordinates.append(rounded_coord)
            filtered_entities.append({"text": entity["text"], "coordinates": rounded_coordinates})
    return filtered_entities

# Function to group by y1 range
def group_by_y1_range(filtered_data, range_threshold=5):
    grouped = []
    temp_group = []
    for item in filtered_data:
        if not temp_group:
            temp_group.append(item)
        else:
            last_y1 = temp_group[-1]['coordinates'][0][1]
            current_y1 = item['coordinates'][0][1]
            if abs(current_y1 - last_y1) <= range_threshold:
                temp_group.append(item)
            else:
                grouped.append(temp_group)
                temp_group = [item]
    if temp_group:
        grouped.append(temp_group)
    return grouped

# Function to sort data by y1 and then x1
def sort_and_group_data(filtered_data):
    data_sorted_by_y1 = sorted(filtered_data, key=lambda x: x['coordinates'][0][1])
    grouped_data = group_by_y1_range(data_sorted_by_y1)
    final_sorted_data = []
    for group in grouped_data:
        sorted_group = sorted(group, key=lambda x: x['coordinates'][0][0])
        final_sorted_data.extend(sorted_group)
    return final_sorted_data

# Function to create the final filtered texts dictionary
def create_filtered_texts(final_sorted_data):
    filtered_texts = {
        "NIK": "", "Nama": "", "Tempat/Tgl Lahir": "", "Jenis kelamin": "", "Gol. Darah": "", "Alamat": "",
        "RT/RW": "", "Kel/Desa": "", "Kecamatan": "", "Agama": "", "Status Perkawinan": "", "Pekerjaan": "",
        "Kewarganegaraan": "", "Berlaku Hingga": ""
    }
    for entity in final_sorted_data:
        x1, y1 = entity['coordinates'][0]
        x2, y2 = entity['coordinates'][1]
        x4, y4 = entity['coordinates'][3]
        if x1 > 210 and x2 < 700 and y1 > 95 and y4 < 154:
            filtered_texts["NIK"] = entity['text']
        if x1 > 225 and x2 < 695 and y1 > 150 and y4 < 199:
            filtered_texts["Nama"] += entity['text'] + " "
        if x1 > 225 and x2 < 715 and y1 > 185 and y4 < 232:
            filtered_texts["Tempat/Tgl Lahir"] += entity['text'] + " "
        if x1 > 225 and x2 < 450 and y1 > 215 and y4 < 259:
            filtered_texts["Jenis kelamin"] += entity['text'] + " "
        if x1 > 615 and x2 < 715 and y1 > 210 and y4 < 264:
            filtered_texts["Gol. Darah"] += entity['text'] + " "
        if x1 > 225 and x2 < 715 and y1 > 240 and y4 < 289:
            filtered_texts["Alamat"] += entity['text'] + " "
        if x1 > 225 and x2 < 390 and y1 > 270 and y4 < 320:
            filtered_texts["RT/RW"] += entity['text'] + " "
        if x1 > 225 and x2 < 510 and y1 > 300 and y4 < 349:
            filtered_texts["Kel/Desa"] += entity['text'] + " "
        if x1 > 225 and x2 < 510 and y1 > 325 and y4 < 384:
            filtered_texts["Kecamatan"] += entity['text'] + " "
        if x1 > 225 and x2 < 400 and y1 > 355 and y4 < 409:
            filtered_texts["Agama"] += entity['text'] + " "
        if x1 > 225 and x2 < 490 and y1 > 385 and y4 < 436:
            filtered_texts["Status Perkawinan"] += entity['text'] + " "
        if x1 > 225 and x2 < 700 and y1 > 415 and y4 < 468:
            filtered_texts["Pekerjaan"] += entity['text'] + " "
        if x1 > 225 and x2 < 355 and y1 > 440 and y4 < 494:
            filtered_texts["Kewarganegaraan"] += entity['text'] + " "
        if x1 > 225 and x2 < 700 and y1 > 475 and y4 < 524:
            filtered_texts["Berlaku Hingga"] += entity['text'] + " "
    for key in filtered_texts:
        filtered_texts[key] = filtered_texts[key].strip().upper()
    return filtered_texts

# Streamlit UI
st.title("OCR KTP Reader")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    image = keras_ocr.tools.read(uploaded_file)
    image_resized = keras_ocr.tools.fit(image, width=1011, height=638, mode="letterbox")

    # Perform OCR
    prediction_groups = pipeline.recognize([image_resized])

    # Draw annotations
    fig, ax = plt.subplots(nrows=1, figsize=(20, 20))
    keras_ocr.tools.drawAnnotations(image=image_resized, predictions=prediction_groups[0], ax=ax)
    st.pyplot(fig)

    # Process OCR results
    ocr_results = []
    for prediction in prediction_groups[0]:
        text, box = prediction
        ocr_results.append({"text": text, "coordinates": box.tolist()})

    # Filter and sort data
    filtered_data = filter_entities(ocr_results)
    final_sorted_data = sort_and_group_data(filtered_data)
    filtered_texts = create_filtered_texts(final_sorted_data)

    # Display JSON results
    json_data = json.dumps(filtered_texts, indent=4)
    st.subheader("OCR Results")
    st.json(json_data)
