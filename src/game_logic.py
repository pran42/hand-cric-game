import random
import numpy as np
from collections import deque

class HandCricketGame:
    def __init__(self):
        self.reset_game()
        
    def reset_game(self):
        # Game state
        self.current_phase = "TOSS"  # TOSS, FIRST_INNINGS, SECOND_INNINGS, GAME_OVER
        self.toss_winner = None
        self.batting_first = None
        self.bowling_first = None
        
        # Scores
        self.first_innings_score = 0
        self.second_innings_score = 0
        self.target_score = 0
        
        # Current players
        self.current_batter = None
        self.current_bowler = None
        
        # Game flags
        self.game_over = False
        self.innings_over = False
        self.last_action = ""
        
        # Toss variables
        self.system_toss_call = None  # 'odd' or 'even'
        self.user_toss_number = None
        self.system_toss_number = None
        self.toss_complete = False
        
        # Current turn
        self.user_number = None
        self.system_number = None
        self.waiting_for_user_input = False
        
        # AI predictor
        self.ai_predictor = SimpleAIPredictor()
        
    def start_toss(self):
        """Start the toss phase - system calls odd or even"""
        if self.system_toss_call is None:
            self.system_toss_call = random.choice(['odd', 'even'])
            self.last_action = f"System calls: {self.system_toss_call.upper()}"
            return "system_called"
        return None
    
    def play_toss(self, user_number):
        """Play the toss with user's number"""
        if self.system_toss_call is None:
            return "System hasn't called yet"
        
        # System shows a random number for toss
        self.system_toss_number = random.randint(1, 6)
        self.user_toss_number = int(user_number)
        
        total = self.user_toss_number + self.system_toss_number
        is_odd = total % 2 == 1
        
        # Determine toss winner
        if (self.system_toss_call == 'odd' and is_odd) or (self.system_toss_call == 'even' and not is_odd):
            self.toss_winner = "system"
        else:
            self.toss_winner = "user"
        
        self.toss_complete = True
        self.last_action = f"Toss: You {self.user_toss_number} + System {self.system_toss_number} = {total} ({'ODD' if is_odd else 'EVEN'}) - {self.toss_winner.upper()} wins!"
        
        return self.toss_winner
    
    def choose_batting_bowling(self, user_choice):
        """User chooses batting or bowling after winning toss"""
        if self.toss_winner != "user":
            return "You didn't win the toss"
        
        user_choice = user_choice.lower()
        if user_choice in ['bat', 'batting']:
            self.batting_first = "user"
            self.bowling_first = "system"
            self.current_batter = "user"
            self.current_bowler = "system"
        elif user_choice in ['bowl', 'bowling']:
            self.batting_first = "system"
            self.bowling_first = "user"
            self.current_batter = "system"
            self.current_bowler = "user"
        else:
            return "Invalid choice. Choose 'bat' or 'bowl'"
        
        self.current_phase = "FIRST_INNINGS"
        self.last_action = f"You chose to {user_choice.upper()} first!"
        return "choice_accepted"
    
    def system_choose_batting_bowling(self):
        """System chooses batting or bowling after winning toss"""
        if self.toss_winner != "system":
            return "System didn't win the toss"
        
        # System randomly chooses
        if random.choice([True, False]):
            self.batting_first = "system"
            self.bowling_first = "user"
            self.current_batter = "system"
            self.current_bowler = "user"
            choice = "BAT"
        else:
            self.batting_first = "user"
            self.bowling_first = "system"
            self.current_batter = "user"
            self.current_bowler = "system"
            choice = "BOWL"
        
        self.current_phase = "FIRST_INNINGS"
        self.last_action = f"System chose to {choice} first!"
        return "system_chose"
    
    def play_ball(self, user_number):
        """Play one ball - both players show numbers"""
        if self.game_over:
            return "GAME_OVER"
        
        self.user_number = int(user_number)
        
        # System chooses its number based on whether it's batting or bowling
        if self.current_batter == "user" and self.current_bowler == "system":
            # User batting, System bowling
            self.system_number = self.ai_predictor.predict_bowling(self.first_innings_score)
        elif self.current_batter == "system" and self.current_bowler == "user":
            # System batting, User bowling
            self.system_number = self.ai_predictor.predict_batting(self.second_innings_score, self.target_score)
        else:
            self.system_number = random.randint(1, 6)
        
        # Check if out (same numbers)
        if self.user_number == self.system_number:
            return self.handle_wicket()
        else:
            return self.handle_runs()
    
    def handle_wicket(self):
        """Handle when batter gets out"""
        if self.current_phase == "FIRST_INNINGS":
            # First innings over
            self.target_score = self.first_innings_score + 1
            self.current_phase = "SECOND_INNINGS"
            
            # Switch roles
            if self.batting_first == "user":
                self.current_batter = "system"
                self.current_bowler = "user"
            else:
                self.current_batter = "user"
                self.current_bowler = "system"
            
            self.innings_over = True
            self.last_action = f"OUT! First innings over. Target: {self.target_score}"
            return "INNINGS_OVER"
        
        else:  # SECOND_INNINGS
            self.game_over = True
            
            if self.second_innings_score == self.target_score - 1:
                # Draw - got out when score equal to first innings
                winner = "DRAW"
                self.last_action = f"OUT! Scores level. Match DRAW!"
            elif self.second_innings_score < self.target_score:
                # Lost - didn't reach target
                winner = "FIRST_BATTER_WINS"
                batter = "User" if self.batting_first == "user" else "System"
                self.last_action = f"OUT! {batter} wins by {self.target_score - self.second_innings_score - 1} runs!"
            else:
                # Shouldn't happen as game should end when target reached
                winner = "UNKNOWN"
            
            return "GAME_OVER"
    
    def handle_runs(self):
        """Handle runs scored"""
        runs_scored = 0
        
        if self.current_batter == "user":
            runs_scored = self.user_number
        else:
            runs_scored = self.system_number
        
        # Add runs to appropriate innings
        if self.current_phase == "FIRST_INNINGS":
            self.first_innings_score += runs_scored
            score = self.first_innings_score
        else:  # SECOND_INNINGS
            self.second_innings_score += runs_scored
            score = self.second_innings_score
            
            # Check if target reached in second innings
            if self.second_innings_score >= self.target_score:
                self.game_over = True
                winner = "SECOND_BATTER_WINS"
                batter = "User" if self.current_batter == "user" else "System"
                self.last_action = f"{batter} wins by {10 - (self.target_score - self.second_innings_score)} wickets!"  # Placeholder
                return "GAME_OVER"
        
        # Update last action
        batter_name = "You" if self.current_batter == "user" else "System"
        bowler_name = "You" if self.current_bowler == "user" else "System"
        
        self.last_action = f"{batter_name}: {self.user_number if self.current_batter == 'user' else self.system_number}, {bowler_name}: {self.system_number if self.current_bowler == 'system' else self.user_number} - Score: {score}"
        
        return "RUNS_SCORED"
    
    def get_system_number_for_display(self):
        """Get system number for animation display"""
        if self.system_number is not None:
            return self.system_number
        
        # For toss display
        if self.current_phase == "TOSS" and self.system_toss_number:
            return self.system_toss_number
        
        # For regular gameplay - predict based on role
        if self.current_batter == "user" and self.current_bowler == "system":
            return self.ai_predictor.predict_bowling(self.first_innings_score)
        elif self.current_batter == "system" and self.current_bowler == "user":
            return self.ai_predictor.predict_batting(self.second_innings_score, self.target_score)
        
        return random.randint(1, 6)
    
    def prepare_system_turn(self):
        """Prepare system's number for display"""
        self.system_number = self.get_system_number_for_display()
        self.waiting_for_user_input = True
        return self.system_number
    
    def get_game_state_description(self):
        """Get description of current game state"""
        if self.current_phase == "TOSS":
            if not self.system_toss_call:
                return "TOSS: System is calling Odd/Even..."
            elif not self.toss_complete:
                return f"TOSS: System calls {self.system_toss_call.upper()} - Show your number!"
            else:
                return f"TOSS: {self.toss_winner.upper()} wins!"
        
        elif self.current_phase == "FIRST_INNINGS":
            batter = "You" if self.current_batter == "user" else "System"
            return f"1st INNINGS: {batter} batting - Score: {self.first_innings_score}"
        
        elif self.current_phase == "SECOND_INNINGS":
            batter = "You" if self.current_batter == "user" else "System"
            return f"2nd INNINGS: {batter} batting - Score: {self.second_innings_score}/{self.target_score}"
        
        elif self.game_over:
            return "GAME OVER"
        
        return "Playing..."

class SimpleAIPredictor:
    def __init__(self):
        self.move_history = deque(maxlen=10)
        
    def predict_bowling(self, current_score):
        """Predict what number to bowl"""
        # Simple strategy: try to get wickets or contain runs
        if current_score == 0:
            return random.randint(1, 6)
        
        # Mix of aggressive and defensive bowling
        if random.random() < 0.3:  # 30% chance to bowl aggressively
            return random.choice([1, 2, 3])  # Lower numbers for variation
        else:
            return random.randint(1, 6)
    
    def predict_batting(self, current_score, target_score):
        """Predict what number to bat while chasing"""
        if target_score == 0:  # First innings
            return random.randint(1, 6)
        
        # Second innings - strategy based on target
        runs_needed = target_score - current_score
        
        if runs_needed <= 3:
            # Need few runs - play safe but ensure runs
            return max(1, min(6, runs_needed))
        elif runs_needed <= 6:
            # Need moderate runs - balanced approach
            return random.randint(max(1, runs_needed - 2), min(6, runs_needed))
        else:
            # Need many runs - aggressive
            return random.randint(3, 6)