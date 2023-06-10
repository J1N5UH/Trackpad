from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLRonPlateau

history = model.fit(
  x_train,
  y_train,
  validation_data=(x_val, y_val),
  epochs=200,
  callbacks=[
    ModelCheckpoint('models/model.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto'),
    ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=50, verbose=1, mode='auto')
  ]
)
