import gradio as gr
import cv2, time


def to_black(image):
    path = "%s.png" % (time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime()))
    cv2.imwrite(
        path,
        image)
    return "success to save " + path


interface = gr.Interface(fn=to_black, inputs="image", outputs='text')
interface.launch(server_name="0.0.0.0", share=True)
