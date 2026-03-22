import cv2
import numpy as np
from pathlib import Path

class NonlinearMotionVideoGenerator:
    """Generate synthetic videos with non-linear object motion"""
    
    def __init__(self, width=640, height=480, fps=30, duration_sec=10):
        self.width = width
        self.height = height
        self.fps = fps
        self.total_frames = int(fps * duration_sec)
        
    def generate_curved_path_video(self, output_file="curved_motion.mp4"):
        """Generate video with curved/circular motion (non-linear)"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, self.fps, 
                             (self.width, self.height))
        
        print(f"🎬 Generating CURVED MOTION video: {output_file}")
        
        for frame_num in range(self.total_frames):
            frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
            
            # Circular motion (non-linear)
            center_x, center_y = self.width // 2, self.height // 2
            radius = 100
            angle = (frame_num / self.total_frames) * 2 * np.pi
            
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))
            
            # Draw moving object
            cv2.circle(frame, (x, y), 20, (0, 0, 255), -1)
            cv2.circle(frame, (x, y), 20, (0, 255, 255), 2)
            
            # Add text
            cv2.putText(frame, f"Frame: {frame_num}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(frame, "Circular Motion (Non-Linear)", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            out.write(frame)
            
            if (frame_num + 1) % 30 == 0:
                print(f"  Frame {frame_num + 1}/{self.total_frames}")
        
        out.release()
        print(f"✅ Created: {output_file}\n")
    
    def generate_zigzag_motion_video(self, output_file="zigzag_motion.mp4"):
        """Generate video with zigzag motion (changes direction abruptly)"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, self.fps, 
                             (self.width, self.height))
        
        print(f"🎬 Generating ZIGZAG MOTION video: {output_file}")
        
        pattern = ["right", "down", "left", "up"]
        pattern_idx = 0
        x, y = 100, self.height // 2
        step_size = 3
        
        for frame_num in range(self.total_frames):
            frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
            
            # Change direction every 75 frames
            if frame_num % 75 == 0 and frame_num > 0:
                pattern_idx = (pattern_idx + 1) % len(pattern)
            
            # Move based on pattern
            direction = pattern[pattern_idx]
            if direction == "right":
                x += step_size
            elif direction == "left":
                x -= step_size
            elif direction == "down":
                y += step_size
            elif direction == "up":
                y -= step_size
            
            # Boundary checking
            x = max(30, min(self.width - 30, x))
            y = max(30, min(self.height - 30, y))
            
            # Draw moving object
            cv2.circle(frame, (x, y), 20, (0, 255, 0), -1)
            cv2.circle(frame, (x, y), 20, (255, 0, 0), 2)
            
            # Add text
            cv2.putText(frame, f"Frame: {frame_num}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(frame, f"Direction: {direction}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            out.write(frame)
            
            if (frame_num + 1) % 30 == 0:
                print(f"  Frame {frame_num + 1}/{self.total_frames}")
        
        out.release()
        print(f"✅ Created: {output_file}\n")
    
    def generate_acceleration_motion_video(self, output_file="acceleration_motion.mp4"):
        """Generate video with acceleration (constant acceleration motion)"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, self.fps, 
                             (self.width, self.height))
        
        print(f"🎬 Generating ACCELERATION MOTION video: {output_file}")
        
        x = 50
        velocity = 1.0
        acceleration = 0.15
        
        for frame_num in range(self.total_frames):
            frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
            
            # Accelerating motion
            velocity += acceleration
            x += velocity
            
            # Bounce back
            if x > self.width - 30:
                x = self.width - 30
                velocity = -5.0
            if x < 30:
                x = 30
                velocity = 5.0
            
            y = self.height // 2
            
            # Draw moving object
            size = int(15 + abs(velocity))  # Size grows with velocity
            cv2.circle(frame, (int(x), y), size, (255, 0, 0), -1)
            cv2.circle(frame, (int(x), y), size, (0, 255, 255), 2)
            
            # Add text
            cv2.putText(frame, f"Frame: {frame_num}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(frame, f"Velocity: {velocity:.2f} | Accel: {acceleration}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            out.write(frame)
            
            if (frame_num + 1) % 30 == 0:
                print(f"  Frame {frame_num + 1}/{self.total_frames}")
        
        out.release()
        print(f"✅ Created: {output_file}\n")
    
    def generate_spiral_motion_video(self, output_file="spiral_motion.mp4"):
        """Generate video with spiral motion (expanding circles)"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, self.fps, 
                             (self.width, self.height))
        
        print(f"🎬 Generating SPIRAL MOTION video: {output_file}")
        
        for frame_num in range(self.total_frames):
            frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
            
            # Spiral motion (expanding radius)
            center_x, center_y = self.width // 2, self.height // 2
            angle = (frame_num / self.total_frames) * 4 * np.pi  # 2 full spirals
            radius = 30 + (frame_num / self.total_frames) * 100
            
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))
            
            # Keep in bounds
            x = max(30, min(self.width - 30, x))
            y = max(30, min(self.height - 30, y))
            
            # Draw moving object
            cv2.circle(frame, (x, y), 15, (0, 165, 255), -1)
            cv2.circle(frame, (x, y), 15, (255, 255, 0), 2)
            
            # Add text
            cv2.putText(frame, f"Frame: {frame_num}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(frame, f"Spiral Motion | Radius: {radius:.1f}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            out.write(frame)
            
            if (frame_num + 1) % 30 == 0:
                print(f"  Frame {frame_num + 1}/{self.total_frames}")
        
        out.release()
        print(f"✅ Created: {output_file}\n")
    
    def generate_multiple_objects_video(self, output_file="multiple_objects.mp4"):
        """Generate video with multiple objects (different motion types)"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, self.fps, 
                             (self.width, self.height))
        
        print(f"🎬 Generating MULTIPLE OBJECTS video: {output_file}")
        
        for frame_num in range(self.total_frames):
            frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
            
            # Object 1: Linear motion
            x1 = 50 + (frame_num * 2) % self.width
            y1 = 80
            cv2.circle(frame, (x1, y1), 20, (255, 0, 0), -1)
            cv2.putText(frame, "Linear", (x1 - 25, y1 - 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
            # Object 2: Circular motion
            angle = (frame_num / self.total_frames) * 2 * np.pi
            x2 = int(self.width // 2 + 80 * np.cos(angle))
            y2 = int(self.height // 2 + 80 * np.sin(angle))
            cv2.circle(frame, (x2, y2), 20, (0, 255, 0), -1)
            cv2.putText(frame, "Circular", (x2 - 30, y2 - 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
            # Object 3: Accelerating motion
            velocity = 1.0 + (frame_num * 0.02)
            x3 = 50 + frame_num * velocity
            y3 = self.height - 100
            x3 = min(self.width - 30, x3)
            cv2.circle(frame, (int(x3), y3), 20, (0, 0, 255), -1)
            cv2.putText(frame, "Accel", (int(x3) - 20, y3 - 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
            # Add text
            cv2.putText(frame, f"Frame: {frame_num}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(frame, "Multiple Objects with Different Motions", 
                       (10, self.height - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 0, 0), 1)
            
            out.write(frame)
            
            if (frame_num + 1) % 30 == 0:
                print(f"  Frame {frame_num + 1}/{self.total_frames}")
        
        out.release()
        print(f"✅ Created: {output_file}\n")


def main():
    print("\n" + "="*80)
    print("🎥 NON-LINEAR MOTION VIDEO DATASET GENERATOR")
    print("="*80 + "\n")
    
    # Create generator
    gen = NonlinearMotionVideoGenerator(width=640, height=480, fps=30, duration_sec=10)
    
    # Generate all videos
    gen.generate_curved_path_video("curved_motion.mp4")
    gen.generate_zigzag_motion_video("zigzag_motion.mp4")
    gen.generate_acceleration_motion_video("acceleration_motion.mp4")
    gen.generate_spiral_motion_video("spiral_motion.mp4")
    gen.generate_multiple_objects_video("multiple_objects.mp4")
    
    print("="*80)
    print("✅ ALL VIDEOS GENERATED SUCCESSFULLY!")
    print("="*80)
    print("\n📊 Generated Videos:")
    print("  1. curved_motion.mp4       - Circular/curved path (perfect for KF)")
    print("  2. zigzag_motion.mp4       - Direction changes (tests prediction)")
    print("  3. acceleration_motion.mp4 - Speed changes (tests velocity)")
    print("  4. spiral_motion.mp4       - Expanding spiral (complex motion)")
    print("  5. multiple_objects.mp4    - Multiple objects (tracking test)")
    print("\n💡 Use these with your Kalman Filter tracker for testing!")
    print("   They're in: c:\\Users\\vansh\\OneDrive\\Desktop\\KALMANFILTER\\")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()