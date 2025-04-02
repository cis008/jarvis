from manim import *

class CreateCircle(Scene):
    def construct(self):
        # Create a circle
        circle = Circle()  
        circle.set_fill(BLUE, opacity=0.5)  # Set color and transparency
        
        # Add the circle to the scene
        self.play(Create(circle))
        self.wait(10)

