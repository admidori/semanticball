from Box2D import (b2World, b2PolygonShape, b2Vec2, b2_staticBody)
from shapely.geometry import Polygon
from shapely.ops import split
import semanticball.lib.pygame as pygame
import cv2

SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480
PPM = 20.0
TIME_STEP = 1.0 / 60
VELOCITY_ITERATIONS = 6
POSITION_ITERATIONS = 2

def create_polygon_shape_from_contour(contour, screen_height):
    if len(contour) < 3:
        return None
    vertices = [(float(point[0][0]), float(point[0][1])) for point in contour]
    vertices = [(x, screen_height - y) for x, y in vertices]
    if len(vertices) > 8:
        step = len(vertices) // 8
        vertices = vertices[::step]
    b2_vertices = [b2Vec2(v[0], v[1]) for v in vertices]
    return b2PolygonShape(vertices=b2_vertices)

def create_static_body_from_contour(contour, world):
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(approx) < 3:
        return
    
    vertices = [(float(point[0][0]), float(point[0][1])) for point in approx]
    vertices = [(SCREEN_WIDTH - x, y) for x,y in vertices]

    b2_vertices = [b2Vec2(v[0] / PPM, v[1] / PPM) for v in vertices]
    
    body = world.CreateDynamicBody()
    if len(b2_vertices) < 3 or len(b2_vertices) > 16:
        return
    shape = b2PolygonShape(vertices=b2_vertices)
    body.CreatePolygonFixture(shape=shape, density=1, friction=0.3, restitution=0.7)

class Ball:
    def __init__(self, world, x, y, radius):
        self.body = world.CreateDynamicBody(position=(x / PPM, y / PPM))
        self.circle = self.body.CreateCircleFixture(radius=radius / PPM, density=1, friction=0.1, restitution=0.7)
        self.radius = radius

    def draw(self, surface):
        position = self.body.position
        position = (int(position.x * PPM), int(SCREEN_HEIGHT - position.y * PPM))
        pygame.draw.circle(surface, (255, 0, 0), position, self.radius)

def detection_vertices(contour,world):
    vertices = [(point[0][0], point[0][1]) for point in contour]
    vertices = [(640 - x, y) for x,y in vertices]
    vertices = [(float(point[0][0]), float(point[0][1])) for point in contour]
    vertices = [(SCREEN_WIDTH - x, y) for x,y in vertices]
    b2_vertices = [b2Vec2(v[0] / PPM, v[1] / PPM) for v in vertices]
    body = world.CreateDynamicBody()
    shape = b2PolygonShape(vertices=b2_vertices)
    if len(b2_vertices) < 3 or len(b2_vertices) > 16:
        print(len(vertices))
        return None
    return body, shape, vertices

def create_world():
    world = b2World(gravity=(0, -10), doSleep=True)
    return world

def step(world):
    world.Step(TIME_STEP, VELOCITY_ITERATIONS, POSITION_ITERATIONS)
