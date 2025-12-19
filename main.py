import pygame
import cv2
import mediapipe as mp
import numpy as np
import random
import time

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 900
FPS = 60
BG_COLOR = (240, 248, 255)
PRIMARY = (30, 144, 255)
SUCCESS = (50, 205, 50)
DANGER = (255, 70, 70)
ACCENT = (255, 165, 0)
WHITE = (255, 255, 255)
DARK_TEXT = (20, 20, 40)


class HandCricketGame:
    """Game logic"""
    
    def __init__(self):
        self.reset_game()
        
    def reset_game(self):
        self.phase = "WELCOME"
        
        # Toss
        self.system_toss_call = None
        self.system_toss_num = None
        self.user_toss_num = None
        self.toss_winner = None
        self.toss_sum = None
        
        # Match
        self.user_batting = True
        self.user_score = 0
        self.system_score = 0
        self.current_batter_runs = 0
        self.target_runs = 0
        
        # Current ball
        self.user_throw = None
        self.system_throw = None
        self.ball_played = False
        self.result_text = ""
        self.awaiting_throw = False
        
        self.round = 1
        self.message = "Welcome to Hand Cricket!"
        
    def start_game(self):
        self.phase = "TOSS_CALL"
        self.system_toss_call = random.choice(['ODD', 'EVEN'])
        self.message = f"System calls: {self.system_toss_call}"
        
    def process_toss(self, user_num):
        self.user_toss_num = int(user_num)
        self.system_toss_num = random.randint(1, 5)  # Changed from 1-6 to 1-5
        self.toss_sum = self.user_toss_num + self.system_toss_num
        is_odd = self.toss_sum % 2 == 1
        
        if (self.system_toss_call == 'ODD' and is_odd) or \
           (self.system_toss_call == 'EVEN' and not is_odd):
            self.toss_winner = "SYSTEM"
            self.message = "SYSTEM WON THE TOSS!"
        else:
            self.toss_winner = "USER"
            self.message = "YOU WON THE TOSS!"
            
        self.phase = "TOSS_RESULT"
        
    def set_batting(self, user_bats):
        self.user_batting = user_bats
        self.phase = "INNINGS_START"
        if user_bats:
            self.message = "YOU ARE BATTING FIRST"
        else:
            self.message = "YOU ARE BOWLING FIRST"
            
    def enable_throw(self):
        """Enable throw mode"""
        self.awaiting_throw = True
        
    def play_ball(self, user_num):
        """Play a ball - count only batter's runs"""
        self.user_throw = int(user_num)
        self.system_throw = random.randint(1, 5)  # Changed from 1-6 to 1-5
        self.ball_played = True
        self.awaiting_throw = False
        
        # Check if wicket (same numbers)
        if self.user_throw == self.system_throw:
            if self.user_batting:
                self.result_text = "YOU ARE OUT!"
            else:
                self.result_text = "SYSTEM IS OUT!"
            self.phase = "WICKET"
        else:
            # Add runs ONLY from batter
            if self.user_batting:
                # User is batter, add only user's throw
                self.current_batter_runs += self.user_throw
                self.user_score = self.current_batter_runs
            else:
                # System is batter, add only system's throw
                self.current_batter_runs += self.system_throw
                self.system_score = self.current_batter_runs
                
            self.result_text = f"RUNS SCORED!"
            self.phase = "BALL_RESULT"
            
            # Check if target reached in round 2
            if self.round == 2 and self.current_batter_runs >= self.target_runs:
                self.phase = "GAME_OVER"
            
    def end_ball(self):
        """Continue or end innings"""
        if self.phase == "WICKET":
            # Batter is out, end innings
            self.phase = "INNINGS_END"
        else:
            # Continue playing
            self.user_throw = None
            self.system_throw = None
            self.ball_played = False
            self.phase = "READY_NEXT_BALL"
            
    def start_second_round(self):
        """Start round 2 - swap roles"""
        self.round += 1
        # Set target as round 1 batter's score + 1
        if self.user_batting:  # If user was batting in round 1
            self.target_runs = self.user_score + 1
        else:  # If system was batting in round 1
            self.target_runs = self.system_score + 1
            
        self.user_batting = not self.user_batting
        self.current_batter_runs = 0
        self.user_throw = None
        self.system_throw = None
        self.ball_played = False
        
        if self.user_batting:
            self.message = f"ROUND 2: YOU ARE BATTING (TARGET: {self.target_runs})"
        else:
            self.message = f"ROUND 2: YOU ARE BOWLING (TARGET: {self.target_runs})"
            
        self.phase = "INNINGS_START"
        
    def determine_winner(self):
        """Determine match winner based on exact rules"""
        if self.round == 2:
            if self.user_batting:  # User was batting in round 2
                if self.user_score >= self.target_runs:
                    return "🎉 YOU WON! (Reached target)"
                else:
                    return "💻 SYSTEM WON"
            else:  # System was batting in round 2
                if self.system_score >= self.target_runs:
                    return "💻 SYSTEM WON! (Reached target)"
                else:
                    return "🎉 YOU WON!"
        return "🤝 IT'S A DRAW!"


class SimpleHandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.last_valid = None
        self.stable_count = 0
        
    def detect_fingers(self, frame):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                tips = [4, 8, 12, 16, 20]
                pips = [2, 6, 10, 14, 18]
                
                raised = 0
                
                if hand.landmark[tips[0]].x < hand.landmark[pips[0]].x:
                    raised += 1
                    
                for i in range(1, 5):
                    if hand.landmark[tips[i]].y < hand.landmark[pips[i]].y:
                        raised += 1
                
                if raised == 0:
                    detected = '1'
                elif raised <= 5:
                    detected = str(raised)
                else:
                    detected = None  # Remove 6 recognition
                
                # Stability check
                if self.last_valid == detected:
                    self.stable_count += 1
                else:
                    self.stable_count = 0
                    self.last_valid = detected
                
                if self.stable_count > 3:
                    return detected
                    
        except:
            pass
        return None


class GameUI:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("🏏 HAND CRICKET")
        self.clock = pygame.time.Clock()
        
        self.font_huge = pygame.font.Font(None, 150)
        self.font_xxl = pygame.font.Font(None, 100)
        self.font_xl = pygame.font.Font(None, 70)
        self.font_lg = pygame.font.Font(None, 50)
        self.font_md = pygame.font.Font(None, 35)
        self.font_sm = pygame.font.Font(None, 25)
        
    def draw_bg(self):
        self.screen.fill(BG_COLOR)
        
    def draw_text(self, text, font, color, x, y):
        text_surf = font.render(str(text), True, color)
        text_rect = text_surf.get_rect(center=(x, y))
        self.screen.blit(text_surf, text_rect)
        
    def render_webcam(self, frame):
        if frame is None:
            return
        try:
            frame = cv2.resize(frame, (300, 225))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rot = np.rot90(frame_rgb)
            surf = pygame.surfarray.make_surface(frame_rot)
            self.screen.blit(surf, (20, 20))
        except:
            pass
        
    def render_welcome(self, game):
        self.draw_bg()
        self.draw_text("🏏", self.font_huge, ACCENT, SCREEN_WIDTH//2, 100)
        self.draw_text("HAND CRICKET", self.font_xxl, PRIMARY, SCREEN_WIDTH//2, 250)
        self.draw_text("Show hand gestures to throw numbers 1-5", self.font_lg, DARK_TEXT, SCREEN_WIDTH//2, 400)  # Changed from 1-6
        self.draw_text("Or use keyboard keys 1-5", self.font_md, PRIMARY, SCREEN_WIDTH//2, 480)  # Changed from 1-6
        self.draw_text("Press SPACE to start", self.font_xl, ACCENT, SCREEN_WIDTH//2, 700)
        self.draw_text("Press Q to quit", self.font_md, (100, 100, 100), SCREEN_WIDTH//2, 800)
        
    def render_toss_call(self, game):
        self.draw_bg()
        self.draw_text("TOSS TIME", self.font_xxl, ACCENT, SCREEN_WIDTH//2, 150)
        self.draw_text(f"System calls: {game.system_toss_call}", self.font_xl, PRIMARY, SCREEN_WIDTH//2, 300)
        self.draw_text("Press SPACE then throw", self.font_lg, DARK_TEXT, SCREEN_WIDTH//2, 500)
        
    def render_throw_mode(self, game, title="THROW NOW!"):
        self.draw_bg()
        self.draw_text(title, self.font_huge, ACCENT, SCREEN_WIDTH//2, 200)
        self.draw_text("Show your number (1-5)", self.font_lg, PRIMARY, SCREEN_WIDTH//2, 500)  # Changed from 1-6
        self.draw_text("Or press key 1-5", self.font_md, DARK_TEXT, SCREEN_WIDTH//2, 600)  # Changed from 1-6
        
    def render_toss_result(self, game):
        self.draw_bg()
        self.draw_text("TOSS RESULT", self.font_xxl, ACCENT, SCREEN_WIDTH//2, 100)
        self.draw_text(f"You threw: {game.user_toss_num}", self.font_lg, PRIMARY, 400, 220)
        self.draw_text(f"System threw: {game.system_toss_num}", self.font_lg, ACCENT, 1000, 220)
        self.draw_text(f"Sum: {game.toss_sum} ({'ODD' if game.toss_sum % 2 == 1 else 'EVEN'})", self.font_xl, SUCCESS, SCREEN_WIDTH//2, 320)
        self.draw_text(game.message, self.font_xl, SUCCESS, SCREEN_WIDTH//2, 420)
        
        if game.toss_winner == "USER":
            self.draw_text("Press B for BAT or O for BOWL", self.font_lg, ACCENT, SCREEN_WIDTH//2, 550)
        else:
            self.draw_text("System choosing...", self.font_lg, PRIMARY, SCREEN_WIDTH//2, 550)
        
    def render_innings_start(self, game):
        self.draw_bg()
        self.draw_text(f"ROUND {game.round}", self.font_xxl, ACCENT, SCREEN_WIDTH//2, 150)
        self.draw_text(game.message, self.font_xl, SUCCESS, SCREEN_WIDTH//2, 300)
        if game.round == 2:
            self.draw_text(f"TARGET: {game.target_runs} RUNS", self.font_lg, PRIMARY, SCREEN_WIDTH//2, 400)
        self.draw_text("Press SPACE to start", self.font_lg, ACCENT, SCREEN_WIDTH//2, 600)
        
    def render_ready_throw(self, game):
        self.draw_bg()
        self.draw_text("READY TO THROW", self.font_xxl, ACCENT, SCREEN_WIDTH//2, 150)
        
        batter = "YOU" if game.user_batting else "SYSTEM"
        self.draw_text(f"{batter} ARE BATTING", self.font_xl, SUCCESS, SCREEN_WIDTH//2, 300)
        self.draw_text(f"Current Score: {game.current_batter_runs}", self.font_lg, PRIMARY, SCREEN_WIDTH//2, 400)
        
        if game.round == 2:
            self.draw_text(f"Target: {game.target_runs} runs", self.font_lg, DANGER, SCREEN_WIDTH//2, 480)
        
        self.draw_text("Press SPACE to throw", self.font_lg, DARK_TEXT, SCREEN_WIDTH//2, 600)
        
    def render_ball_result(self, game):
        self.draw_bg()
        # Show numbers thrown
        self.draw_text(f"You threw: {game.user_throw}", self.font_lg, PRIMARY, 400, 200)
        self.draw_text(f"System threw: {game.system_throw}", self.font_lg, ACCENT, 1000, 200)
        
        # Show who is batter and what counts
        batter = "YOU" if game.user_batting else "SYSTEM"
        batter_number = game.user_throw if game.user_batting else game.system_throw
        self.draw_text(f"Batter ({batter}) scored: {batter_number} runs!", self.font_xl, SUCCESS, SCREEN_WIDTH//2, 350)
        
        self.draw_text(f"Total Runs: {game.current_batter_runs}", self.font_lg, PRIMARY, SCREEN_WIDTH//2, 450)
        
        if game.round == 2:
            remaining = game.target_runs - game.current_batter_runs
            if remaining > 0:
                self.draw_text(f"Need {remaining} more runs", self.font_md, DANGER, SCREEN_WIDTH//2, 520)
            else:
                self.draw_text("TARGET REACHED!", self.font_md, SUCCESS, SCREEN_WIDTH//2, 520)
        
        self.draw_text("Press SPACE to continue", self.font_md, DARK_TEXT, SCREEN_WIDTH//2, 620)
        
    def render_wicket(self, game):
        self.draw_bg()
        # Show numbers thrown
        self.draw_text(f"You threw: {game.user_throw}", self.font_lg, PRIMARY, 400, 200)
        self.draw_text(f"System threw: {game.system_throw}", self.font_lg, ACCENT, 1000, 200)
        
        self.draw_text("WICKET!", self.font_xxl, DANGER, SCREEN_WIDTH//2, 350)
        self.draw_text(game.result_text, self.font_xl, DANGER, SCREEN_WIDTH//2, 450)
        self.draw_text("Same numbers = Batter is OUT!", self.font_lg, DARK_TEXT, SCREEN_WIDTH//2, 550)
        self.draw_text("Press SPACE to continue", self.font_lg, ACCENT, SCREEN_WIDTH//2, 650)
        
    def render_innings_end(self, game):
        self.draw_bg()
        self.draw_text(f"ROUND {game.round} ENDED", self.font_xxl, ACCENT, SCREEN_WIDTH//2, 150)
        
        if game.user_batting:
            self.draw_text(f"Your Score: {game.user_score}", self.font_xl, SUCCESS, SCREEN_WIDTH//2, 300)
            target = game.user_score + 1
            self.draw_text(f"TARGET FOR SYSTEM: {target} RUNS", self.font_lg, PRIMARY, SCREEN_WIDTH//2, 400)
        else:
            self.draw_text(f"System Score: {game.system_score}", self.font_xl, SUCCESS, SCREEN_WIDTH//2, 300)
            target = game.system_score + 1
            self.draw_text(f"TARGET FOR YOU: {target} RUNS", self.font_lg, PRIMARY, SCREEN_WIDTH//2, 400)
        
        self.draw_text("Press SPACE for ROUND 2", self.font_lg, ACCENT, SCREEN_WIDTH//2, 550)
        
    def render_game_over(self, game):
        self.draw_bg()
        self.draw_text("GAME OVER", self.font_xxl, ACCENT, SCREEN_WIDTH//2, 150)
        self.draw_text(game.determine_winner(), self.font_xl, SUCCESS, SCREEN_WIDTH//2, 300)
        
        # Show final scores
        if game.round == 1:
            round1_batter = "YOU" if game.user_batting else "SYSTEM"
            round1_score = game.user_score if game.user_batting else game.system_score
            self.draw_text(f"Round 1 ({round1_batter} batting): {round1_score} runs", self.font_lg, DARK_TEXT, SCREEN_WIDTH//2, 400)
        else:
            round1_score = game.system_score if game.user_batting else game.user_score
            round2_score = game.user_score if game.user_batting else game.system_score
            round1_batter = "SYSTEM" if game.user_batting else "YOU"
            round2_batter = "YOU" if game.user_batting else "SYSTEM"
            
            self.draw_text(f"Round 1 ({round1_batter}): {round1_score} runs", self.font_md, DARK_TEXT, SCREEN_WIDTH//2, 380)
            self.draw_text(f"Round 2 ({round2_batter}): {round2_score} runs", self.font_md, DARK_TEXT, SCREEN_WIDTH//2, 430)
            self.draw_text(f"Target: {game.target_runs} runs", self.font_md, PRIMARY, SCREEN_WIDTH//2, 480)
        
        self.draw_text("Press SPACE to play again | Q to quit", self.font_lg, ACCENT, SCREEN_WIDTH//2, 600)


class HandCricketApp:
    def __init__(self):
        self.ui = GameUI()
        self.game = HandCricketGame()
        self.ui.render_welcome(self.game)
        
        self.cap = None
        self.camera_available = False
        self.init_camera()
        
        self.detector = SimpleHandDetector()
        self.running = True
        self.last_detection = 0
        self.current_frame = None
        
    def init_camera(self):
        try:
            self.cap = cv2.VideoCapture(0)
            ret, frame = self.cap.read()
            if ret:
                self.camera_available = True
        except:
            pass
            
    def process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 'QUIT'
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_q, pygame.K_ESCAPE]:
                    return 'QUIT'
                if pygame.K_1 <= event.key <= pygame.K_5:  # Changed from K_6 to K_5
                    return str(event.key - pygame.K_0)
                if event.key == pygame.K_b:
                    return 'BAT'
                if event.key == pygame.K_o:
                    return 'BOWL'
                if event.key == pygame.K_SPACE:
                    return 'SPACE'
        return None
        
    def get_webcam_frame(self):
        if not self.cap or not self.camera_available:
            return None
        try:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = cv2.flip(frame, 1)
                return self.current_frame
        except:
            pass
        return None
        
    def detect_hand(self):
        if self.game.awaiting_throw and time.time() - self.last_detection > 0.3:
            frame = self.current_frame
            if frame is not None:
                num = self.detector.detect_fingers(frame)
                self.last_detection = time.time()
                return num
        return None
        
    def run(self):
        auto_timer = 0
        
        while self.running:
            frame = self.get_webcam_frame()
            inp = self.process_events()
            
            if inp == 'QUIT':
                self.running = False
                break
            
            # Detect hand
            hand_num = self.detect_hand()
            if hand_num and hand_num in ['1', '2', '3', '4', '5']:  # Removed '6'
                inp = hand_num
            
            # Game states
            if self.game.phase == "WELCOME":
                self.ui.render_welcome(self.game)
                if frame is not None:
                    self.ui.render_webcam(frame)
                if inp == 'SPACE':
                    self.game.start_game()
                    
            elif self.game.phase == "TOSS_CALL":
                self.ui.render_toss_call(self.game)
                if frame is not None:
                    self.ui.render_webcam(frame)
                if inp == 'SPACE':
                    self.game.enable_throw()
                    self.game.phase = "TOSS_THROW"
                    self.last_detection = time.time()
                    
            elif self.game.phase == "TOSS_THROW":
                self.ui.render_throw_mode(self.game, "THROW FOR TOSS")
                if frame is not None:
                    self.ui.render_webcam(frame)
                if inp and inp in ['1', '2', '3', '4', '5']:  # Removed '6'
                    self.game.process_toss(inp)
                    
            elif self.game.phase == "TOSS_RESULT":
                self.ui.render_toss_result(self.game)
                if frame is not None:
                    self.ui.render_webcam(frame)
                if self.game.toss_winner == "USER":
                    if inp == 'BAT':
                        self.game.set_batting(True)
                    elif inp == 'BOWL':
                        self.game.set_batting(False)
                else:
                    auto_timer += 1
                    if auto_timer > 90:
                        self.game.set_batting(random.choice([True, False]))
                        auto_timer = 0
                        
            elif self.game.phase == "INNINGS_START":
                self.ui.render_innings_start(self.game)
                if frame is not None:
                    self.ui.render_webcam(frame)
                if inp == 'SPACE':
                    self.game.phase = "READY_NEXT_BALL"
                    
            elif self.game.phase == "READY_NEXT_BALL":
                self.ui.render_ready_throw(self.game)
                if frame is not None:
                    self.ui.render_webcam(frame)
                if inp == 'SPACE':
                    self.game.enable_throw()
                    self.game.phase = "THROW_MODE"
                    self.last_detection = time.time()
                    
            elif self.game.phase == "THROW_MODE":
                self.ui.render_throw_mode(self.game)
                if frame is not None:
                    self.ui.render_webcam(frame)
                if inp and inp in ['1', '2', '3', '4', '5']:  # Removed '6'
                    self.game.play_ball(inp)
                    
            elif self.game.phase == "BALL_RESULT":
                self.ui.render_ball_result(self.game)
                if frame is not None:
                    self.ui.render_webcam(frame)
                if inp == 'SPACE':
                    self.game.end_ball()
                    
            elif self.game.phase == "WICKET":
                self.ui.render_wicket(self.game)
                if frame is not None:
                    self.ui.render_webcam(frame)
                if inp == 'SPACE':
                    self.game.end_ball()
                    
            elif self.game.phase == "INNINGS_END":
                self.ui.render_innings_end(self.game)
                if frame is not None:
                    self.ui.render_webcam(frame)
                if inp == 'SPACE':
                    if self.game.round == 1:
                        self.game.start_second_round()
                    else:
                        self.game.phase = "GAME_OVER"
                    
            elif self.game.phase == "GAME_OVER":
                self.ui.render_game_over(self.game)
                if frame is not None:
                    self.ui.render_webcam(frame)
                if inp == 'SPACE':
                    self.game.reset_game()
                    self.ui.render_welcome(self.game)
            
            pygame.display.flip()
            self.ui.clock.tick(FPS)
        
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()


if __name__ == "__main__":
    app = HandCricketApp()
    app.run()