import os
import math
import numpy as np
import pandas as pd
import pydicom as dicom
import matplotlib.pyplot as plt
import cv2

TF_AVAILABLE = True
try:
    import tensorflow._api.v2.compat.v1 as tf
    tf.disable_v2_behavior()
except Exception:
    tf = None
    TF_AVAILABLE = False

from sklearn.metrics import confusion_matrix

from tkinter import *
from tkinter import messagebox, ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import tkinter.font as tkFont


class LCD_CNN:
    def __init__(self,root):
        self.root=root
        #window size
        self.root.geometry("1006x500+0+0")
        self.root.resizable(False, False)
        self.root.title("Lung Cancer Detection")

        # Define custom fonts
        self.title_font = tkFont.Font(family="Helvetica", size=24, weight="bold")
        self.button_font = tkFont.Font(family="Arial", size=12, weight="bold")
        self.label_font = tkFont.Font(family="Arial", size=10, weight="bold")

        # Define styles
        style = ttk.Style()
        style.configure("TButton", font=self.button_font, padding=6, relief="flat", background="#4CAF50", foreground="white")
        style.map("TButton", background=[("active", "#45a049")])
        style.configure("TLabel", font=self.label_font, background="#f0f0f0", foreground="#333")
        style.configure("TFrame", background="#f0f0f0")

        # Create a medical-themed background with text overlay
        try:
            # Try to load a medical scan image first
            img4=Image.open(os.path.join('Images', 'ct_scan.jpg'))
            img4=img4.resize((1006,500),Image.ANTIALIAS)
        except Exception:
            try:
                img4=Image.open(os.path.join('Images', 'medical_scan.jpg'))
                img4=img4.resize((1006,500),Image.ANTIALIAS)
            except Exception:
                try:
                    img4=Image.open(os.path.join('Images', 'lung_cancer_bg.jpg'))
                    img4=img4.resize((1006,500),Image.ANTIALIAS)
                except Exception:
                    try:
                        img4=Image.open(os.path.join('Images', 'Lung-Cancer-Detection.jpg'))
                        img4=img4.resize((1006,500),Image.ANTIALIAS)
                    except Exception:
                        # Create a medical-themed background with gradient and text
                        img4 = Image.new('RGB', (1006, 500), color='#1e3a5f')
                        draw = ImageDraw.Draw(img4)

                        # Create gradient effect
                        for y in range(500):
                            r = int(30 + (y / 500) * 50)
                            g = int(58 + (y / 500) * 70)
                            b = int(95 + (y / 500) * 80)
                            draw.line((0, y, 1006, y), fill=(r, g, b))

                        # Add medical symbols
                        try:
                            # Use default font
                            font = ImageFont.load_default()
                        except:
                            font = None

                        # Draw medical cross
                        draw.rectangle([450, 200, 556, 300], fill='#ffffff', outline='#ff0000', width=3)
                        draw.rectangle([493, 157, 513, 343], fill='#ffffff', outline='#ff0000', width=3)

                        # Add text
                        if font:
                            draw.text((400, 350), "LUNG CANCER DETECTION SYSTEM", fill='#ffffff', font=font)
                            draw.text((450, 380), "Advanced AI-Powered Analysis", fill='#cccccc', font=font)

        self.photoimg4=ImageTk.PhotoImage(img4)

        # Background image
        bg_img=Label(self.root,image=self.photoimg4)
        bg_img.place(x=0,y=50,width=1006,height=500)

        # Title Label with modern styling
        title_frame = ttk.Frame(self.root, style="TFrame")
        title_frame.place(x=0, y=0, width=1006, height=50)
        title_lbl=ttk.Label(title_frame, text="Lung Cancer Detection", font=self.title_font, background="#2196F3", foreground="white", anchor="center")
        title_lbl.pack(fill="both", expand=True)

        # Main frame for buttons and content
        main_frame = ttk.Frame(self.root, style="TFrame")
        main_frame.place(x=50, y=100, width=300, height=350)

        # Progress bar
        self.progress = ttk.Progressbar(main_frame, orient="horizontal", length=280, mode="determinate")
        self.progress.pack(pady=10)

        # Buttons with icons (using text icons for simplicity, can replace with images)
        self.b1=ttk.Button(main_frame, text="📁 Import Data", command=self.import_data, style="TButton")
        self.b1.pack(pady=5, fill="x")

        self.b2=ttk.Button(main_frame, text="⚙️ Pre-Process Data", command=self.preprocess_data, style="TButton", state="disabled")
        self.b2.pack(pady=5, fill="x")

        self.b3=ttk.Button(main_frame, text="🧠 Train Data", command=self.train_data, style="TButton", state="disabled")
        self.b3.pack(pady=5, fill="x")

        self.b4=ttk.Button(main_frame, text="🔍 Detect Cancer", command=self.detect_cancer, style="TButton", state="disabled")
        self.b4.pack(pady=5, fill="x")

        # Info frame for displaying stats
        info_frame = ttk.Frame(self.root, style="TFrame")
        info_frame.place(x=700, y=100, width=250, height=200)
        ttk.Label(info_frame, text="Training Info", font=self.label_font).pack(pady=5)
        self.training_data_label = ttk.Label(info_frame, text="")
        self.training_data_label.pack(pady=2)
        self.validation_data_label = ttk.Label(info_frame, text="")
        self.validation_data_label.pack(pady=2)
        self.final_accuracy_label = ttk.Label(info_frame, text="")
        self.final_accuracy_label.pack(pady=2)

        # Add medical icons or images to the interface
        try:
            # Load small medical icon
            icon_img = Image.open(os.path.join('Images', 'ct_scan.jpg'))
            icon_img = icon_img.resize((50, 50), Image.ANTIALIAS)
            self.icon_photo = ImageTk.PhotoImage(icon_img)
            icon_label = ttk.Label(main_frame, image=self.icon_photo)
            icon_label.pack(pady=10)
        except Exception:
            pass  # Skip if icon loading fails

#Data Import lets you upload data from external sources and combine it with data you collect via Analytics.
    def import_data(self):
        ##Data directory
        default_dir = 'Datasetssss'
        if os.path.isdir(default_dir):
            self.dataDirectory = default_dir
        else:
            # prompt user to select the data directory
            from tkinter import filedialog
            sel = filedialog.askdirectory(title='Select dataset folder (contains patient subfolders)')
            if not sel:
                messagebox.showerror('Import Data', 'No data directory selected. Import cancelled.')
                return
            self.dataDirectory = sel

        try:
            self.lungPatients = [d for d in os.listdir(self.dataDirectory) if os.path.isdir(os.path.join(self.dataDirectory, d))]
        except Exception as e:
            messagebox.showerror('Import Data', f'Could not read data directory: {e}')
            return

        ##Read labels csv 
        try:
            self.labels = pd.read_csv('stage1_labels.csv', index_col=0)
        except Exception as e:
            messagebox.showwarning('Import Data', f"Could not read 'stage1_labels.csv': {e}\nYou can still preprocess, but labeled results may be missing.")
            self.labels = pd.DataFrame()

        # Attempt to match labels to patient folders using DICOM metadata (PatientID, StudyInstanceUID, AccessionNumber)
        self.matched_labels = {}
        try:
            self._match_labels_by_dicom()
        except Exception as e:
            print('Label matching by DICOM failed:', e)

        ##Setting x*y size to 10
        self.size = 10

        ## Setting z-dimension (number of slices to 5)
        self.NoSlices = 5

        messagebox.showinfo("Import Data" , f"Data Imported Successfully!\nFound {len(self.lungPatients)} patient folders")

        self.b1["state"] = "disabled"
        self.b2["state"] = "normal"

# Data preprocessing is the process of transforming raw data into an understandable format.
    def preprocess_data(self):

        def chunks(l, n):
            count = 0
            for i in range(0, len(l), n):
                if (count < self.NoSlices):
                    yield l[i:i + n]
                    count = count + 1


        def mean(l):
            return sum(l) / len(l)
        #Average


        def dataProcessing(patient, labels_df, size=10, noslices=5, visualize=False):
            # read label if available: prefer matched label from DICOM, then CSV lookup by folder name
            label = None
            try:
                if hasattr(self, 'matched_labels') and patient in self.matched_labels:
                    label = self.matched_labels.get(patient)
                elif not labels_df.empty:
                    label = labels_df._get_value(patient, 'cancer')
            except Exception:
                label = None

            path = os.path.join(self.dataDirectory, patient)
            # read files in folder, attempt to load DICOMs, skip invalid files
            files = sorted(os.listdir(path))
            slices = []
            for fname in files:
                full = os.path.join(path, fname)
                try:
                    ds = dicom.dcmread(full)
                except Exception:
                    try:
                        # try force-reading if header is missing
                        ds = dicom.dcmread(full, force=True)
                    except Exception:
                        continue
                # ensure we have pixel data
                if not hasattr(ds, 'pixel_array'):
                    continue
                slices.append(ds)

            # sort slices: prefer ImagePositionPatient Z, then InstanceNumber, else filename order
            def _slice_z(s):
                try:
                    return float(s.ImagePositionPatient[2])
                except Exception:
                    try:
                        return float(s.InstanceNumber)
                    except Exception:
                        return 0.0

            slices.sort(key=_slice_z)

            # Resize slices to (size, size)
            resized_slices = [cv2.resize(np.array(each_slice.pixel_array), (size, size)) for each_slice in slices]

            # If no valid slices, create zero-frames
            if len(resized_slices) == 0:
                new_slices = [np.zeros((size, size), dtype=np.float32) for _ in range(noslices)]
            else:
                # Split the list of slices into `noslices` groups as evenly as possible
                chunks_list = np.array_split(resized_slices, noslices)
                new_slices = []
                for chunk in chunks_list:
                    # chunk may be an empty array when there are fewer slices than noslices
                    if getattr(chunk, 'size', 0) == 0:
                        new_slices.append(np.zeros((size, size), dtype=np.float32))
                    else:
                        # stack and average across the chunk (axis=0) to produce one (size,size) frame
                        stack = np.stack(chunk, axis=0).astype(np.float32)
                        mean_img = np.mean(stack, axis=0)
                        new_slices.append(mean_img)

            if label == 1: #Cancer Patient
                label = [0, 1]
            elif label == 0:    #Non Cancerous Patient
                label = [1, 0]
            else:
                label = [0, 0]  # Unknown/unlabeled
            return np.array(new_slices), label


        imageData = []
        #Check if Data Labels is available in CSV or not
        for num, patient in enumerate(self.lungPatients):
            if num % 50 == 0:
                print('Saved -', num)
            try:
                img_data, label = dataProcessing(patient, self.labels, size=self.size, noslices=self.NoSlices)
                imageData.append([img_data, label,patient])
            except Exception as e:
                print(f'Error processing {patient}: {e}')

        print(f'Total patients processed: {len(imageData)}')

        ##Results= Image Data and lable.
        # Build an object array so np.save doesn't try to coerce a heterogeneous list
        try:
            arr = np.empty(len(imageData), dtype=object)
            for i, v in enumerate(imageData):
                arr[i] = v
            np.save('imageDataNew-{}-{}-{}.npy'.format(self.size, self.size, self.NoSlices), arr, allow_pickle=True)
        except Exception as e:
            print('Failed to save imageData:', e)
            raise

        messagebox.showinfo("Pre-Process Data", f"Data Pre-Processing Done Successfully!\nProcessed {len(imageData)} patients")

        self.b2["state"] = "disabled"
        self.b3["state"] = "normal"

# Data training is the process of training the model based on the dataset and then predict on new data.
    def train_data(self):    
        # Load preprocessed data
        expected_fname = 'imageDataNew-{}-{}-{}.npy'.format(10, 10, 5)
        if not os.path.exists(expected_fname):
            messagebox.showerror("Train Data", "No preprocessed data found.\nPlease run 'Pre-Process Data' first.")
            return

        try:
            imageData = np.load(expected_fname, allow_pickle=True)
        except Exception as e:
            messagebox.showerror("Train Data", f"Failed to load preprocessed data: {e}")
            return

        # Ensure we have a list of samples: each item should be [img_array, label, patient]
        try:
            imageData = list(imageData)
        except Exception:
            imageData = [imageData]

        total_samples = len(imageData)
        print(f"Total samples in imageData: {total_samples}")

        if total_samples < 2:
            messagebox.showerror("Train Data", f"Not enough data to train. Found {total_samples} samples; need at least 2.")
            return

        # Normalize labels helper
        def _label_to_scalar(lab):
            try:
                if isinstance(lab, (list, tuple, np.ndarray)) and len(lab) == 2:
                    # assume [no_cancer, cancer]
                    return int(lab[1] == 1)
                if isinstance(lab, (int, float)):
                    return int(lab)
                # if it's a string like '0' or '1'
                return int(str(lab))
            except Exception:
                return 0

        # Build X/y full lists and keep patient ids
        X = []
        y = []
        patients = []
        for item in imageData:
            try:
                img = np.array(item[0]).astype(np.float32)
                lbl = item[1]
                pat = item[2] if len(item) > 2 else None
            except Exception:
                # unexpected format, skip
                continue
            X.append(img.ravel())
            y.append(_label_to_scalar(lbl))
            patients.append(pat)

        # Split into training/validation (80/20)
        split_idx = max(1, int(0.8 * len(X)))
        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_val = X[split_idx:]
        y_val = y[split_idx:]

        if len(X_val) == 0 and len(X_train) > 1:
            # ensure at least 1 val sample
            X_val = X_train[-1:]
            y_val = y_train[-1:]
            X_train = X_train[:-1]
            y_train = y_train[:-1]

        self.training_data_label.config(text=f"Training Data: {len(X_train)}")
        self.validation_data_label.config(text=f"Validation Data: {len(X_val)}")

        # If TF not available, use scikit-learn fallback
        if not TF_AVAILABLE:
            try:
                from sklearn.linear_model import LogisticRegression
            except Exception as e:
                messagebox.showwarning("Train Data", f"TensorFlow not available and scikit-learn import failed: {e}\nInstall scikit-learn or TensorFlow to enable training.")
                return

            if len(X_train) == 0 or len(X_val) == 0:
                messagebox.showwarning("Train Data", f"Not enough data for fallback training. Train: {len(X_train)}, Val: {len(X_val)}. Need at least 1 sample in each.")
                return

            if len(X_train) < 3:
                messagebox.showwarning("Train Data", f"Very small dataset ({len(X_train)} training samples). Results may not be reliable.")

            clf = LogisticRegression(solver='liblinear', max_iter=1000)
            # Handle case where all labels are the same (common when CSV labels don't match folders)
            try:
                unique_labels = set(y_train)
            except Exception:
                unique_labels = set()

            if len(unique_labels) <= 1:
                # derive demo labels from mean intensity across available samples
                try:
                    messagebox.showwarning("Train Data", "All provided labels are identical or missing. Deriving demo labels from image intensity for training.")
                except Exception:
                    print("Warning: all labels identical; deriving demo labels from intensity.")
                all_X = X_train + X_val
                if len(all_X) == 0:
                    messagebox.showerror("Train Data", "No data available for deriving labels.")
                    return
                means = [float(np.mean(x)) for x in all_X]
                med = float(np.median(means))
                # assign label 1 if mean > median, else 0
                try:
                    y_train = [1 if float(np.mean(x)) > med else 0 for x in X_train]
                    if len(X_val) > 0:
                        y_val = [1 if float(np.mean(x)) > med else 0 for x in X_val]
                except Exception:
                    # fallback: keep original labels and let fit fail
                    pass

            try:
                # ensure there are at least two classes
                if len(set(y_train)) <= 1:
                    messagebox.showerror("Train Data", "Training aborted: labels contain only one class even after attempting a demo-label derivation.")
                    return
                clf.fit(X_train, y_train)
            except Exception as e:
                messagebox.showerror("Train Data", f"Fallback training failed: {e}")
                return

            # Score
            try:
                acc = clf.score(X_val, y_val) if len(X_val) > 0 else clf.score(X_train, y_train)
            except Exception:
                acc = 0.0

            # Save model
            import pickle
            try:
                with open('trained_model.pkl', 'wb') as f:
                    pickle.dump(clf, f)
                print("Model saved to 'trained_model.pkl'")
            except Exception as e:
                print(f"Warning: Could not save model: {e}")

            self.final_accuracy_label.config(text=f"Final Accuracy: {acc:.4f}")

            # Display predictions and confusion matrix where possible
            try:
                preds = clf.predict(X_val)
                cm = confusion_matrix(y_val, preds)
                print('Confusion Matrix:\n', cm)
            except Exception:
                preds = []

            messagebox.showinfo("Train Data", "Fallback training completed (scikit-learn).")
            self.b3["state"] = "disabled"
            self.b4["state"] = "normal"
            return

        # TensorFlow branch (existing network code expects trainingData/validationData format)
        # Recreate `trainingData` and `validationData` lists from the original `imageData`
        trainingData = []
        validationData = []
        for idx, item in enumerate(imageData):
            try:
                imgarr = np.array(item[0]).astype(np.float32)
                lbl = item[1]
                # ensure one-hot label
                if isinstance(lbl, (list, tuple, np.ndarray)) and len(lbl) == 2:
                    onehot = lbl
                else:
                    onehot = [0, 1] if _label_to_scalar(lbl) == 1 else [1, 0]
                entry = [imgarr, onehot, item[2] if len(item) > 2 else None]
            except Exception:
                continue
            if idx < split_idx:
                trainingData.append(entry)
            else:
                validationData.append(entry)

        if len(validationData) == 0 and len(trainingData) > 1:
            validationData = trainingData[-1:]
            trainingData = trainingData[:-1]

        # Now proceed to TensorFlow placeholders
        x = tf.placeholder('float')
        y = tf.placeholder('float')
        size = 10
        keep_rate = 0.8
        NoSlices = 5

        def convolution3d(x, W):
            return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')


        def maxpooling3d(x):
            return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

        def cnn(x):
            x = tf.reshape(x, shape=[-1, size, size, NoSlices, 1])
            convolution1 = tf.nn.relu(
                convolution3d(x, tf.Variable(tf.random_normal([3, 3, 3, 1, 32]))) + tf.Variable(tf.random_normal([32])))
            convolution1 = maxpooling3d(convolution1)
            convolution2 = tf.nn.relu(
                convolution3d(convolution1, tf.Variable(tf.random_normal([3, 3, 3, 32, 64]))) + tf.Variable(
                    tf.random_normal([64])))
            convolution2 = maxpooling3d(convolution2)
            convolution3 = tf.nn.relu(
                convolution3d(convolution2, tf.Variable(tf.random_normal([3, 3, 3, 64, 128]))) + tf.Variable(
                    tf.random_normal([128])))
            convolution3 = maxpooling3d(convolution3)
            convolution4 = tf.nn.relu(
                convolution3d(convolution3, tf.Variable(tf.random_normal([3, 3, 3, 128, 256]))) + tf.Variable(
                    tf.random_normal([256])))
            convolution4 = maxpooling3d(convolution4)
            convolution5 = tf.nn.relu(
                convolution3d(convolution4, tf.Variable(tf.random_normal([3, 3, 3, 256, 512]))) + tf.Variable(
                    tf.random_normal([512])))
            convolution5 = maxpooling3d(convolution5)
            fullyconnected = tf.reshape(convolution5, [-1, 256])
            fullyconnected = tf.nn.relu(
                tf.matmul(fullyconnected, tf.Variable(tf.random_normal([256, 256]))) + tf.Variable(tf.random_normal([256])))
            fullyconnected = tf.nn.dropout(fullyconnected, keep_rate)
            output = tf.matmul(fullyconnected, tf.Variable(tf.random_normal([256, 2]))) + tf.Variable(tf.random_normal([2]))
            return output

        def network(x):
            prediction = cnn(x)
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)
            epochs = 100
            with tf.Session() as session:
                session.run(tf.global_variables_initializer())
                for epoch in range(epochs):
                    epoch_loss = 0
                    for data in trainingData:
                        try:
                            X = data[0]
                            Y = data[1]
                            _, c = session.run([optimizer, cost], feed_dict={x: X, y: Y})
                            epoch_loss += c
                        except Exception as e:
                            pass
                        
                    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
                   # if tf.argmax(prediction, 1) == 0:
                    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
                    print('Epoch', epoch + 1, 'completed out of', epochs, 'loss:', epoch_loss)
                    # print('Correct:',correct.eval({x:[i[0] for i in validationData], y:[i[1] for i in validationData]}))
                    print('Accuracy:', accuracy.eval({x: [i[0] for i in validationData], y: [i[1] for i in validationData]}))
                #print('Final Accuracy:', accuracy.eval({x: [i[0] for i in validationData], y: [i[1] for i in validationData]}))
                x1 = accuracy.eval({x: [i[0] for i in validationData], y: [i[1] for i in validationData]})

                self.final_accuracy_label.config(text=f"Final Accuracy: {x1:.4f}")

                patients = []
                actual = []
                predicted = []

                finalprediction = tf.argmax(prediction, 1)
                actualprediction = tf.argmax(y, 1)
                for i in range(len(validationData)):
                    patients.append(validationData[i][2])
                for i in finalprediction.eval({x: [i[0] for i in validationData], y: [i[1] for i in validationData]}):
                    if(i==1):
                        predicted.append("Cancer")
                    else:
                        predicted.append("No Cancer")
                for i in actualprediction.eval({x: [i[0] for i in validationData], y: [i[1] for i in validationData]}):
                    if(i==1):
                        actual.append("Cancer")
                    else:
                        actual.append("No Cancer")
                for i in range(len(patients)):
                    print("----------------------------------------------------")
                    print("Patient: ",patients[i])
                    print("Actual: ", actual[i])
                    print("Predicted: ", predicted[i])
                    print("----------------------------------------------------")

                # messagebox.showinfo("Result" , "Patient: " + ' '.join(map(str,patients)) + "\nActual: " + str(actual) + "\nPredicted: " + str(predicted) + "Accuracy: " + str(x1))    

                y_actual = pd.Series(
                    (actualprediction.eval({x: [i[0] for i in validationData], y: [i[1] for i in validationData]})),
                    name='Actual')
                y_predicted = pd.Series(
                    (finalprediction.eval({x: [i[0] for i in validationData], y: [i[1] for i in validationData]})),
                    name='Predicted')

                df_confusion = pd.crosstab(y_actual, y_predicted).reindex(columns=[0,1],index=[0,1], fill_value=0)
                print('Confusion Matrix:\n')
                print(df_confusion)

                prediction_label=ttk.Label(self.root, text=">>>>    P R E D I C T I O N    <<<<", font=self.label_font, background="#FF9800", foreground="white", anchor="center")
                prediction_label.place(x=0,y=458,width=1006,height=20)

                result1 = []

                for i in range(len(validationData)):
                    result1.append(patients[i])
                    if(y_actual[i] == 1):
                        result1.append("Cancer")
                    else:
                        result1.append("No Cancer")

                    if(y_predicted[i] == 1):
                        result1.append("Cancer")
                    else:
                        result1.append("No Cancer")

                # print(result1)

                total_rows = int(len(patients))
                total_columns = int(len(result1)/len(patients))  

                heading = ["Patient: ", "Actual: ", "Predicted: "]

                self.root.geometry("1006x"+str(500+(len(patients)*20)-20)+"+0+0")
                self.root.resizable(False, False)

                for i in range(total_rows):
                    for j in range(total_columns):
                        self.e = Entry(self.root, width=42, fg='black', font=self.label_font)
                        self.e.place(x=(j*335),y=(478+i*20))
                        self.e.insert(END, heading[j] + result1[j + i*3])
                        self.e["state"] = "disabled"

                self.b3["state"] = "disabled"
                self.b4["state"] = "normal"

                messagebox.showinfo("Train Data" , "Model Trained Successfully!")

                ## Function to plot confusion matrix
                def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):\

                    plt.matshow(df_confusion, cmap=cmap)  # imshow  
                    # plt.title(title)
                    plt.colorbar()
                    tick_marks = np.arange(len(df_confusion.columns))
                    plt.title(title)
                    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
                    plt.yticks(tick_marks, df_confusion.index)
                    # plt.tight_layout()
                    plt.ylabel(df_confusion.index.name)
                    plt.xlabel(df_confusion.columns.name)
                    plt.show()
                plot_confusion_matrix(df_confusion)
                # print(y_true,y_pred)
                # print(confusion_matrix(y_true, y_pred))
                # print(actualprediction.eval({x:[i[0] for i in validationData], y:[i[1] for i in validationData]}))
                # print(finalprediction.eval({x:[i[0] for i in validationData], y:[i[1] for i in validationData]}))  

        network(x)

        # Also train & persist a small scikit-learn model for detection convenience
        try:
            from sklearn.linear_model import LogisticRegression
            import pickle
            if len(X) > 0 and len(y) > 0:
                clf2 = LogisticRegression(solver='liblinear', max_iter=1000)
                clf2.fit(X, y)
                with open('trained_model.pkl', 'wb') as f:
                    pickle.dump(clf2, f)
                print("Saved fallback scikit model to 'trained_model.pkl' for detection.")
        except Exception as e:
            print('Could not train/save scikit fallback model:', e)

# Cancer Detection Mode: Load trained model and predict on new image
    def detect_cancer(self):
        """Load a patient DICOM folder and predict cancer risk"""
        from tkinter import filedialog
        import pickle
        
        # Select a patient folder to analyze
        patient_folder = filedialog.askdirectory(title='Select patient folder (contains DICOM slices)')
        if not patient_folder:
            messagebox.showwarning('Detect Cancer', 'No folder selected.')
            return
        
        # Extract patient name from path
        patient_name = os.path.basename(patient_folder)
        
        # Read and preprocess the DICOM slices
        try:
            files = sorted(os.listdir(patient_folder))
            slices = []
            for fname in files:
                full = os.path.join(patient_folder, fname)
                try:
                    ds = dicom.dcmread(full)
                except Exception:
                    try:
                        ds = dicom.dcmread(full, force=True)
                    except Exception:
                        continue
                if not hasattr(ds, 'pixel_array'):
                    continue
                slices.append(ds)
            
            if len(slices) == 0:
                messagebox.showerror('Detect Cancer', 'No valid DICOM files found in the folder.')
                return
            
            # Sort slices
            def _slice_z(s):
                try:
                    return float(s.ImagePositionPatient[2])
                except Exception:
                    try:
                        return float(s.InstanceNumber)
                    except Exception:
                        return 0.0
            slices.sort(key=_slice_z)
            
            # Preprocess: resize and chunk slices (same as training)
            size = 10
            noslices = 5
            processed_slices = [cv2.resize(np.array(s.pixel_array), (size, size)) for s in slices]

            # Build `noslices` averaged frames using numpy array_split (robust for few slices)
            if len(processed_slices) == 0:
                new_slices = [np.zeros((size, size), dtype=np.float32) for _ in range(noslices)]
            else:
                chunks_list = np.array_split(processed_slices, noslices)
                new_slices = []
                for chunk in chunks_list:
                    if getattr(chunk, 'size', 0) == 0:
                        new_slices.append(np.zeros((size, size), dtype=np.float32))
                    else:
                        stack = np.stack(chunk, axis=0).astype(np.float32)
                        mean_img = np.mean(stack, axis=0)
                        new_slices.append(mean_img)

            processed_image = np.array(new_slices)
            
        except Exception as e:
            messagebox.showerror('Detect Cancer', f'Error processing DICOM files: {e}')
            return
        
        # Try to load trained scikit-learn model
        model_path = 'trained_model.pkl'
        if os.path.exists(model_path):
            try:
                import pickle
                with open(model_path, 'rb') as f:
                    clf = pickle.load(f)
                
                # Flatten and predict
                X_test = processed_image.ravel().reshape(1, -1)
                pred = clf.predict(X_test)[0]
                pred_proba = clf.predict_proba(X_test)[0]
                
                result = "Cancer" if pred == 1 else "No Cancer"
                confidence = pred_proba[1] if pred == 1 else pred_proba[0]
                
                msg = f"Patient: {patient_name}\n\nPrediction: {result}\nConfidence: {confidence*100:.2f}%"
                messagebox.showinfo('Detection Result', msg)
                print(msg)
                
            except Exception as e:
                messagebox.showerror('Detect Cancer', f'Error loading/using model: {e}\nTrain a model first.')
        else:
            messagebox.showwarning('Detect Cancer', 'No trained model found.\nPlease train the model first (click "Train Data").')

# For GUI
if __name__ == "__main__":
        root=Tk()
        obj=LCD_CNN(root)
        root.mainloop()
