# Name of the tensor for feeding the input image as jpeg.
tensor_name_input_jpeg = "DecodeJpeg/contents:0"

# Name of the tensor for feeding the decoded input image.
# Use this for feeding images in other formats than jpeg.
tensor_name_input_image = "DecodeJpeg:0"

# Name of the tensor for the resized input image.
# This is used to retrieve the image after it has been resized.
tensor_name_resized_image = "ResizeBilinear:0"

# Name of the tensor for the output of the softmax-classifier.
# This is used for classifying images with the Inception model.
tensor_name_softmax = "softmax:0"

# Name of the tensor for the unscaled outputs of the softmax-classifier (aka. logits).
tensor_name_softmax_logits = "softmax/logits:0"

# Name of the tensor for the output of the Inception model.
# This is used for Transfer Learning.
tensor_name_transfer_layer = "pool_3:0"