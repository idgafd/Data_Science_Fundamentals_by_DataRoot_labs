from env.core.snake import Snake


def full_test_snake_step(test_head_coordinates, test_direction_index, test_snake_length):

    def test_steps(test_snake, test_head_coordinates, test_snake_length):
        try:
            # Snake moves RIGHT, try to press RIGHT
            snake_position_init = test_snake.blocks.copy()
            expected_new_head_1st_step = (test_head_coordinates[0], test_head_coordinates[1] + 1)
            expected_tail_1st_step = (test_head_coordinates[0], test_head_coordinates[1] - test_snake_length + 1)
            new_head_1st_step, tail_1st_step = test_snake.step(1)

            assert new_head_1st_step == expected_new_head_1st_step, '\nNew Snake head is created incorrectly' \
                                                                    f'\nInitial Snake position: {snake_position_init}' \
                                                                    '\nInitial head direction: 1' \
                                                                    '\nUsing snake.step(1)' \
                                                                    f'\nNow Your snake looks like: {test_snake.blocks}'\
                                                                    f'\nYour new Snake head: {new_head_1st_step}' \
                                                                    f'\nExpected: {expected_new_head_1st_step}'

            assert tail_1st_step == expected_tail_1st_step, '\nNew Snake tail is created incorrectly' \
                                                            f'\nInitial Snake position: {snake_position_init}' \
                                                            f'\nInitial head direction: 1' \
                                                            '\nUsing snake.step(1)' \
                                                            f'\nNow Your snake looks like: {test_snake.blocks}' \
                                                            f'\nYour old Snake tail: {tail_1st_step}' \
                                                            f'\nExpected: {expected_tail_1st_step}'

            # Snake moves RIGHT, try to press LEFT
            snake_position_1st = test_snake.blocks.copy()
            expected_new_head_2nd_step = (new_head_1st_step[0], new_head_1st_step[1] + 1)
            expected_tail_2nd_step = (tail_1st_step[0], tail_1st_step[1] + 1)
            new_head_2nd_step, tail_2nd_step = test_snake.step(3)

            assert new_head_2nd_step == expected_new_head_2nd_step, '\nNew Snake head is created incorrectly' \
                                                                    f'\nInitial Snake position: {snake_position_1st}' \
                                                                    '\nInitial head direction: 1' \
                                                                    '\nUsing snake.step(3)' \
                                                                    f'\nNow Your snake looks like: {test_snake.blocks}' \
                                                                    f'\nYour new Snake head: {new_head_2nd_step}' \
                                                                    f'\nExpected: {expected_new_head_2nd_step}'

            assert tail_2nd_step == expected_tail_2nd_step, '\nNew Snake tail is created incorrectly' \
                                                            f'\nInitial Snake position: {snake_position_1st}' \
                                                            f'\nInitial head direction: 1' \
                                                            '\nUsing snake.step(3)' \
                                                            f'\nNow Your snake looks like: {test_snake.blocks}' \
                                                            f'\nYour old Snake tail: {tail_2nd_step}' \
                                                            f'\nExpected: {expected_tail_2nd_step}'

            # Snake moves RIGHT, try to press DOWN
            snake_position_2nd = test_snake.blocks.copy()
            expected_new_head_3rd_step = (new_head_2nd_step[0] + 1, new_head_2nd_step[1])
            expected_tail_3rd_step = (tail_2nd_step[0], tail_2nd_step[1] + 1)
            new_head_3rd_step, tail_3rd_step = test_snake.step(2)

            assert new_head_3rd_step == expected_new_head_3rd_step, '\nNew Snake head is created incorrectly' \
                                                                    f'\nInitial Snake position: {snake_position_2nd}' \
                                                                    '\nInitial head direction: 1' \
                                                                    '\nUsing snake.step(2)' \
                                                                    f'\nNow Your snake looks like: {test_snake.blocks}' \
                                                                    f'\nYour new Snake head: {new_head_3rd_step}' \
                                                                    f'\nExpected: {expected_new_head_3rd_step}'

            assert tail_3rd_step == expected_tail_3rd_step, '\nNew Snake tail is created incorrectly' \
                                                            f'\nInitial Snake position: {snake_position_2nd}' \
                                                            f'\nInitial head direction: 1' \
                                                            '\nUsing snake.step(3)' \
                                                            f'\nNow Your snake looks like: {test_snake.blocks}' \
                                                            f'\nYour old Snake tail: {tail_2nd_step}' \
                                                            f'\nExpected: {expected_tail_2nd_step}'

            # Snake moves DOWN, try to press UP
            snake_position_3rd = test_snake.blocks.copy()
            expected_new_head_4th_step = (new_head_3rd_step[0] + 1, new_head_3rd_step[1])
            expected_tail_4th_step = (tail_3rd_step[0], tail_3rd_step[1] + 1)
            new_head_4th_step, tail_4th_step = test_snake.step(0)

            assert new_head_4th_step == expected_new_head_4th_step, '\nNew Snake head is created incorrectly' \
                                                                    f'\nInitial Snake position: {snake_position_3rd}' \
                                                                    '\nInitial head direction: 1' \
                                                                    '\nUsing snake.step(2)' \
                                                                    f'\nNow Your snake looks like: {test_snake.blocks}' \
                                                                    f'\nYour new Snake head: {new_head_4th_step}' \
                                                                    f'\nExpected: {expected_new_head_4th_step}'

            assert tail_4th_step == expected_tail_4th_step, '\nNew Snake tail is created incorrectly' \
                                                            f'\nInitial Snake position: {snake_position_3rd}' \
                                                            f'\nInitial head direction: 1' \
                                                            '\nUsing snake.step(3)' \
                                                            f'\nNow Your snake looks like: {test_snake.blocks}' \
                                                            f'\nYour old Snake tail: {tail_4th_step}' \
                                                            f'\nExpected: {expected_tail_4th_step}'
            return 0
        except AssertionError as e:
            print(e)
            return 1

    def test_length(test_snake, test_snake_length):
        try:
            assert len(test_snake.blocks) == test_snake_length, f'\nWrong length of the Snake:' \
                                                          f'\nExpected: {test_snake_length}' \
                                                          f'\nBut your is: {len(test_snake.blocks)}'
            return 0
        except AssertionError as e:
            print(e)
            return 1

    def test_types(test_snake):
        try:
            assert type(test_snake.blocks) == list, f'\nWrong type of snake.blocks:' \
                                                    f'\nIt should be list' \
                                                    f'\nBut your is {type(test_snake.blocks)}'

            for coordinates in test_snake.blocks:
                assert type(coordinates) == tuple, f'\nWrong type of coordinates:' \
                                                   f'\nThey all should be tuples' \
                                                   f'\nBut you have: {type(coordinates)}'

            return 0
        except AssertionError as e:
            print(e)
            return 1

    test_snake = Snake(test_head_coordinates, test_direction_index, test_snake_length)

    step_error_occurred = test_steps(test_snake, test_head_coordinates, test_snake_length)
    length_error_occurred = test_length(test_snake, test_snake_length)
    types_error_occurred = test_types(test_snake)

    if step_error_occurred or length_error_occurred or types_error_occurred:
        return 0
    else:
        return 1