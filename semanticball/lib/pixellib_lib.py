"""
This source code is modify from pixellib library.
The original source code is available at https://github.com/ayoolaolafenwa/PixelLib
And the original source code is under MIT License.

    MIT License

    Copyright (c) 2020 ayoolaolafenwa

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

from pixellib.torchbackend.instance import instanceSegmentation
import os
import colorsys
import random
import semanticball.lib.opencv_lib as opencv_lib
import numpy as np
import cv2
import pygame
import semanticball.lib.box2d_lib as box2d_lib
import datetime

class overrideInstaceSegmentation(instanceSegmentation):
    def __init__(self):
        super().__init__()

    def desplay_box_instance(image, boxes, masks, class_ids, class_name, scores, show_bboxes, text_size, box_thickness, text_thickness):
        n_instances = boxes.shape[0]
        colors = overrideInstaceSegmentation.random_colors(n_instances)

        txt_color = (255,255,255)
        for i, color in enumerate(colors):
            mask = masks[:,:,i]
            color = (1.0,0.0,0.0)
            image = overrideInstaceSegmentation.apply_mask(image, mask, color)
            if not np.any(boxes[i]):
                continue

            if show_bboxes == True:    
                x1, y1, x2, y2 = boxes[i]
                label = class_name[class_ids[i]]
            
                score = scores[i] if scores is not None else None
            
                caption = '{} {:.2f}'.format(label, score) if score else label
            
            
                color_rec = [int(c) for c in np.array(colors[i]) * 255]
                image = cv2.rectangle(image, (x1, y1), (x2, y2), color_rec, box_thickness)
                image = cv2.putText(image, caption, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, text_size,  txt_color, text_thickness)
            
        return image
    
    def process_camera(self, cam, show_bboxes = False, segment_target_classes = None, extract_segmented_objects = False,
    extract_from_box = False,save_extracted_objects = False,  text_thickness = 1,text_size = 0.6, box_thickness = 2,
    mask_points_values = False, output_video_name = None, frames_per_second = None,
    show_frames = None, frame_name = None, verbose = None, check_fps = False):

        capture = cam
        width, height = opencv_lib.edge_cutout(capture)

        if output_video_name is not None:
            codec = cv2.VideoWriter_fourcc(*'DIVX')
            save_video = cv2.VideoWriter(output_video_name, codec, frames_per_second, (width, height))

        counter = 0
        start = datetime.now()     

        world = box2d_lib.create_world()

        pygame.init()
        screen_width = 640
        screen_height = 480
        screen = pygame.display.set_mode((screen_width, screen_height))
        clock = pygame.time.Clock()

        # ボールのリスト
        balls = []
        ball_radius = 10  # ボールの半径を20ピクセルに設定
        ball_spawn_interval = 5
        frame_count = 0

        while True:
            frame_count += 1
            if frame_count % ball_spawn_interval == 0:
                ball = box2d_lib.Ball(world, random.randint(ball_radius, screen_width - ball_radius), screen_height-100, ball_radius)
                balls.append(ball)

            ret, frame = capture.read()
            counter += 1
            if ret:
                seg, output =  self.segmentFrame(frame, show_bboxes=show_bboxes, segment_target_classes=segment_target_classes,
                        text_thickness = text_thickness,text_size = text_size, box_thickness = box_thickness,
                        extract_segmented_objects=extract_segmented_objects, extract_from_box = extract_from_box,
                        save_extracted_objects=save_extracted_objects,
                        mask_points_values= mask_points_values)
                output = cv2.resize(output, (width,height), interpolation=cv2.INTER_AREA)

                if show_frames == True:
                    if frame_name is not None:
                        contours = opencv_lib.capture_edge(output)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = np.rot90(frame)
                        frame = pygame.surfarray.make_surface(frame)
                        screen.blit(frame, (0, 0))

                        for contour in contours:
                            body, shape, vertices = box2d_lib.detection_vertices(contour, world)
                            if body is None:
                                continue
                            if len(vertices) > 2:
                                pygame.draw.polygon(screen, (0, 255, 0), vertices, 2)

                            body.CreatePolygonFixture(shape=shape, density=1, friction=0.1, restitution=0.1)

                        #for body in world.bodies:
                        #    if body.type == b2_staticBody:
                        #        world.DestroyBody(body)

                        for ball in balls:
                            ball.draw(screen)
                        
                        box2d_lib.step(world)
                        pygame.display.flip()

                        if cv2.waitKey(25) & 0xFF == ord('q'):
                            break
                if output_video_name is not None:
                    save_video.write(output)
            elif counter == 30:
                break  

        end = datetime.now() 
            
        if check_fps == True:
            timetaken = (end-start).total_seconds()
            fps = counter/timetaken
            print(round(fps), "frames per second")   

        if verbose is not None:
            print(f"Processed {counter} frames in {timetaken:.1f} seconds") 

        capture.release()

        if output_video_name is not None:
            save_video.release()  

        return seg, output

    def random_colors(N, bright=True):
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors

    def apply_mask(image, mask, color, alpha=0.5):
        alpha = 1
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                    image[:, :, c] *
                                    (1 - alpha) + alpha * color[c] * 255,
                                    image[:, :, c])
        return image
def segment():
    capture = cv2.VideoCapture(0)
    segment_video = overrideInstaceSegmentation()
    
    if os.path.exists("model/pointrend_resnet50.pkl"):
        segment_video.load_model("pointrend_resnet50.pkl", confidence=0.7, detection_speed="fast")
    else:
        print("Model not found.")
        return None
    
    segment_video.load_model("pointrend_resnet50.pkl", confidence=0.7, detection_speed="fast")
    segment_video.process_camera(capture, frames_per_second=15, show_frames=True, frame_name="frame", mask_points_values=True)
