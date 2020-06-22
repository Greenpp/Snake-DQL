from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label

from game import Engine


class SnakeGame(Widget):
    def __init__(self, **kwargs):
        self.score_label = kwargs.pop('score_label', None)
        super(SnakeGame, self).__init__(**kwargs)
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)

        self.engine = Engine()
        self.block_size = 10
        self.board_length = (self.block_size + 1) * self.engine.board_size

    def update(self, dt):
        if self.engine.alive:
            self.engine.next_round()
            self.engine.check_if_alive()
            self.draw_board()
            self.update_score()

    def update_score(self):
        score = self.engine.points
        self.score_label.text = f'Points: {score}'

    def draw_board(self):
        self.canvas.clear()

        with self.canvas:
            border_width = 5
            self.padding_x = (self.width - self.board_length) // 2
            self.padding_y = (self.height - self.board_length) // 2
            Rectangle(pos=(self.padding_x - border_width, self.padding_y),
                      size=(border_width, self.board_length))
            Rectangle(pos=(self.padding_x, self.padding_y - border_width),
                      size=(self.board_length, border_width))
            Rectangle(pos=(self.padding_x + self.board_length,
                           self.padding_y), size=(border_width, self.board_length))
            Rectangle(pos=(self.padding_x, self.padding_y +
                           self.board_length), size=(self.board_length, border_width))

            Color(.59, .91, .12)
            for s in self.engine.snake:
                x, y = s
                Rectangle(pos=(self.padding_x + x * (self.block_size + 1), self.padding_y +
                               y * (self.block_size + 1)), size=(self.block_size, self.block_size))
            x, y = self.engine.fruit
            Color(.93, .83, .05)
            Rectangle(pos=(self.padding_x + x * (self.block_size + 1), self.padding_y +
                           y * (self.block_size + 1)), size=(self.block_size, self.block_size))

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if keycode[1] == 'w':
            self.engine.change_direction('up')
        elif keycode[1] == 's':
            self.engine.change_direction('down')
        elif keycode[1] == 'a':
            self.engine.change_direction('left')
        elif keycode[1] == 'd':
            self.engine.change_direction('right')
        elif keycode[1] == 'r':
            self.engine = Engine()


class SnakeApp(App):
    def build(self):
        label = Label(text='Halo', size_hint=(.3, 1), font_size='30sp')
        game = SnakeGame(score_label=label)
        Clock.schedule_interval(game.update, game.engine.round_time)

        layout = GridLayout(cols=2, padding=[20])
        layout.add_widget(game)
        layout.add_widget(label)

        return layout


if __name__ == '__main__':
    SnakeApp().run()
