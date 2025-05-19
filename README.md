# NEURAL-STYLE-TRANSFER 
*CAMPANY* : CODTECH IT SOLUTION 

*NAME* :Kasala Suvarna Nandini

*INTERN ID* : CODF56

*DOMAIN* : Artifical Intelligence  

*DURATION* : 4 WEEKS 

*MENTOR*: NEELA SANTOSH 

## Project Describtion :
This project focuses on implementing a Neural Style Transfer (NST) system with an interactive Streamlit-based web interface, allowing users to blend the content of one image with the artistic style of another. The core idea behind NST is to generate a new image that visually resembles the content image but adopts the artistic characteristics of the style image. This technique is based on a deep convolutional neural network (CNN), specifically the VGG19 model pretrained on ImageNet.

The system works by extracting deep features from both the content and style images. These features are obtained from intermediate layers of the VGG19 model. The content features represent the high-level structure and objects in the image, while the style features capture textures, patterns, and colors. To replicate style, the project computes the Gram matrix of style features, which reflects the spatial correlation between different feature maps. During optimization, the target image is iteratively updated using gradient descent to minimize a loss function that combines content loss (difference between content features) and style loss (difference between Gram matrices).

To make the project accessible and interactive, it integrates Streamlit, a Python library for building web applications. Users can upload their own content and style images directly from the browser. Once the images are uploaded, the model runs the style transfer algorithm for a set number of iterations (typically around 200–300) and displays the resulting stylized image on the interface.

The application does not require any hardcoded file paths; instead, it uses file upload widgets and displays the uploaded and generated images on the same page. This design makes the app user-friendly and ideal for educational demos, artistic experiments, or real-time creative applications.

From a technical standpoint, the app uses PyTorch for neural network operations and model handling. The VGG19 model is used only as a feature extractor and is kept in evaluation mode without updating its parameters. The target image starts as a clone of the content image and is optimized using the Adam optimizer. The model’s performance depends on system resources — it can run on both CPU and GPU, although GPU is recommended for faster results.

This project combines the power of deep learning with a simple UI, making it a practical example of how machine learning can be integrated into web applications. It serves as a foundation for more advanced applications, such as real-time style transfer in video, mobile apps, or augmented reality.

In summary, this Neural Style Transfer project bridges the gap between complex AI models and user-friendly interaction. It demonstrates how deep learning models like VGG19 can be repurposed for artistic creation, and how frameworks like Streamlit can bring them to life through intuitive interfaces. This blend of machine learning, computer vision, and user experience design makes it a compelling and modern AI project.
## Output : 
![Image](https://github.com/user-attachments/assets/ecbe2bb1-d4a0-431f-b142-5409f19e6ae6) 
