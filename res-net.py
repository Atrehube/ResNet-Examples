import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pathlib
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
import csv
import shutil  
import os  
  

def assemble_data(img_height,img_width):
    # Segment data as desired using file manipulation
    dataset_url = "http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_png.zip"
    data_dir = tf.keras.utils.get_file(origin=dataset_url, 
                                       fname='sketches', 
                                       extract=True)

    data_dir = pathlib.Path(data_dir[:-8]+'png')

    #image_count = len(list(data_dir.glob('*/*.png')))
    labels = [str(i)[len(str(data_dir))+1:] for i in list(data_dir.glob('*'))]
    labels.remove('filelist.txt')
    try:
        os.makedirs(str(data_dir)+'_val')
        for label in labels:
            old_dir = str(data_dir) +'\\'+label
            new_dir = str(data_dir) +'_val\\'+label
            os.mkdir(new_dir)
            files = os.listdir(old_dir)
            maxlen = len(max(files, key=len))
            files = [(' ' * (maxlen - len(x))) + x for x in files]
            files.sort()
            files = [x.strip(' ') for x in files]
            while len(files) > 60:
                to_move = files.pop()
                source = old_dir + '\\' + to_move
                dest = new_dir + '\\' + to_move
                shutil.move(source,dest)
    except:
        pass
        
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
          data_dir,
          image_size=(img_height, img_width),
          batch_size=batch_size)
        
            
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
          str(data_dir) +'_val',
          image_size=(img_height, img_width),
          batch_size=batch_size)

    
    AUTOTUNE = tf.data.AUTOTUNE
    
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds,labels


def record(history, name, save_model):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig(name)
    plt.close()
    
    if save_model:
        scratch_model.save(name)
    
    with open(name+".csv", "w") as outfile:
       writer = csv.writer(outfile, lineterminator = '\n')
       writer.writerow(history.history.keys())
       writer.writerows(zip(*history.history.values()))
       
if __name__ == '__main__':
    # Prohibit memory exhausting GPU operations. 
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    # Image and batch size
    batch_size = 32
    img_height = 224
    img_width = 224
    
         
    # Collect data 
    train_ds, val_ds, labels = assemble_data(img_height,img_width)
    num_classes = len(labels)
    # Built in resNet50 architecture
    resNet50 = tf.keras.applications.ResNet50(
        include_top=True,
        weights= None,
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=num_classes)
    # Some data augmentation layers
    data_augmentation = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.2),
    ])
    # Model from scrath
    scratch_model = tf.keras.Sequential([
        data_augmentation,
        layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1),
        resNet50
         ])
    # Compile
    scratch_model.compile(
      optimizer='adam',
      loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'])
    # 20 epochs proof of concept
    epochs = 2
    # fit model
    history = scratch_model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=epochs
    )
    # Call funtion to record output
    record(history,'scratch',False)
    
    # Begin problem 2
    
    
    IMG_SHAPE = (img_width, img_height) + (3,)
    preprocess_input = layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)
    
    # Gather resnet50 model with imagenet weights
    resNet50Trained = tf.keras.applications.ResNet50(
        include_top=False,
        weights= 'imagenet',
        input_tensor=None,
        input_shape=IMG_SHAPE,
        pooling=None)
    
    image_batch, label_batch = next(iter(train_ds))
    feature_batch = resNet50Trained(image_batch)
    
    resNet50Trained.trainable = False
    # Make sure to specify training = False when finetuning the model.
    
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    
    prediction_layer = tf.keras.layers.Dense(num_classes)
    prediction_batch = prediction_layer(feature_batch_average)
    
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = resNet50Trained(x, training=False)
    x = global_average_layer(x)
    #x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    
    #base_learning_rate = 0.0001
    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    #model.summary()
    
    initial_epochs = 2 # HARD cap. 
    
    # Stop new epochs when the model has converged by using callbacks
    usualCallback = EarlyStopping()
    overfitCallback = EarlyStopping(monitor='loss', min_delta=0, patience = 20)
    
    #loss0, accuracy0 = model.evaluate(val_ds)
    #print("initial loss: {:.2f}".format(loss0))
    #print("initial accuracy: {:.2f}".format(accuracy0))
    
    history = model.fit(train_ds,
                        epochs=initial_epochs,
                        validation_data=val_ds,
                        callbacks=[overfitCallback])
    
    record(history,'fixedFeatureExtractor',False)
    
    # Problem 3
    
    
    resNet50Trained = tf.keras.applications.ResNet50(
        include_top=False,
        weights= 'imagenet',
        input_tensor=None,
        input_shape=IMG_SHAPE,
        pooling=None)
    
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = resNet50Trained(x, training=False)
    x = global_average_layer(x)
    #x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    
    resNet50Trained.trainable = True
    print("Number of layers in the base model: ", len(resNet50Trained.layers))
    
    # Fine-tune from this layer onwards
    fine_tune_at = 50
    # Because I only wanted to fine tune the higher levels. 
    
    # Freeze all the layers before the `fine_tune_at` layer
    for layer in resNet50Trained.layers[:fine_tune_at]:
      layer.trainable =  False
    
    model.compile(loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
                  metrics=['accuracy'])
    
    model.summary()
    
    history_fine = model.fit(train_ds,
                        epochs=initial_epochs,
                        validation_data=val_ds,
                        callbacks=[overfitCallback])
    record(history_fine,'Finetuned',False)
       
    # Competition
    
    
    batch_size = 32
    img_height = 299
    img_width = 299
    
    train_ds, val_ds, labels = assemble_data(img_height,img_width) 
    
    inceptionRes = tf.keras.applications.InceptionResNetV2(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape= (299,299,3)
        #classes=len(labels),
    )
    
    #preprocess_input = tf.keras.applications.inception_resnet_v2.preprocess_input
    
    
    inputs = tf.keras.Input(shape=(299, 299, 3) )
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = inceptionRes.output
    #x = inceptionRes(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dense(2048, activation='relu')(x)
    #x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inceptionRes.input, outputs)
    
    inceptionRes.trainable = True
    print("Number of layers in the base model: ", len(inceptionRes.layers))
    
    # Fine-tune from this layer onwards
    fine_tune_at = 390
    # Because I only wanted to fine tune the higher levels. 
    
    # Freeze all the layers before the `fine_tune_at` layer
    for i in range(len(inceptionRes.layers)):
        if i < fine_tune_at:
            inceptionRes.layers[i].trainable =  False
        else:
            inceptionRes.layers[i].trainable =  True
            
            
    model.compile(loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
                  metrics=['accuracy'])
    
    model.summary()
    
    history_fine = model.fit(train_ds,
                        epochs=initial_epochs,
                        validation_data=val_ds,
                        callbacks=[overfitCallback])
    
    record(history_fine,'competition',False)
