import unittest

import project.bot.parsers as parser


class ParserTests(unittest.TestCase):
    def setUp(self) -> None:
        return None

    def tearDown(self) -> None:
        pass

    def test_parse_year_function(self):
        input_values = [
            2007,
            '2007',
            '2007-2009',
            'Продам 2007',
            None
        ]
        output_values = [
            2007,
            2007,
            2008,
            2007,
            None
        ]
        for input_value, output_value in zip(input_values, output_values):
            self.assertEquals(parser.parse_year(input_value), output_value)

    def test_parse_price_function(self):
        input_values = [
            '32000$',
            '32000 $',
            ' 32000 $',
            32000,
            None
        ]
        output_values = [
            32000,
            32000,
            32000,
            32000,
            None
        ]
        for input_value, output_value in zip(input_values, output_values):
            self.assertEquals(parser.parse_price(input_value), output_value)

    def test_parse_verification_function(self):
        input_values = [
            ['Перевірена ціна', 'Перевірена квартира'],
            ['Перевірена ціна'],
            ['Перевірена квартира'],
            [],
            None
        ]
        output_values = [
            (True, True),
            (True, False),
            (False, True),
            (False, False),
            (False, False)
        ]
        for input_value, output_value in zip(input_values, output_values):
            self.assertEquals(parser.parse_verification(input_value), output_value)

    def test_parse_images_function(self):
        input_values = [
            [
                'https://cdn.riastatic.com/photos/dom/panoramas/154/15436/1543609/1543609i.jpg?v=14214',
                'https://cdn.riastatic.com/photos/dom/panoramas/154/15436/1543611/1543611i.jpg?v=14214'
            ],
            [
                'https://cdn.riastatic.com/photos/dom/panoramas/154/15436/1543610/1543610i.jpg?v=14214'
            ],
            [
                'https://cdn.riastatic.com/docs/dom/support/3/c0/19e7093a922357cd3ab95ed42cbf2.jpg'
            ],
            [],
            [
                'https://cdn.riastatic.com/photos/dom/panoramas/154/15436/1543608/1543608i.jpg?v=14214',
                'https://cdn.riastatic.com/photos/dom/panoramas/154/15436/1543612/1543612i.jpg?v=14214',
                'https://cdn.riastatic.com/photosnew/dom/photo/perevireno-prodaja-kvartira-vinnitsa-akademicheskiy-mikolayivska-ulitsa__100813798m.jpg',
                'https://cdn.riastatic.com/photosnew/dom/photo/perevireno-prodaja-kvartira-vinnitsa-akademicheskiy-mikolayivska-ulitsa__100814247m.jpg',
                'https://cdn.riastatic.com/photosnew/dom/photo/perevireno-prodaja-kvartira-vinnitsa-akademicheskiy-mikolayivska-ulitsa__100814244m.jpg',
                'https://cdn.riastatic.com/docs/dom/support/3/c0/19e7093a922357cd3ab95ed42cbf2.jpg'
            ]
        ]
        output_values = [
            [
                'https://cdn.riastatic.com/photos/dom/panoramas/154/15436/1543609/1543609m.jpg?v=14214',
                'https://cdn.riastatic.com/photos/dom/panoramas/154/15436/1543611/1543611m.jpg?v=14214'
            ],
            [
                'https://cdn.riastatic.com/photos/dom/panoramas/154/15436/1543610/1543610m.jpg?v=14214',
            ],
            None,
            None,
            [
                'https://cdn.riastatic.com/photos/dom/panoramas/154/15436/1543608/1543608m.jpg?v=14214',
                'https://cdn.riastatic.com/photos/dom/panoramas/154/15436/1543612/1543612m.jpg?v=14214',
                'https://cdn.riastatic.com/photosnew/dom/photo/perevireno-prodaja-kvartira-vinnitsa-akademicheskiy-mikolayivska-ulitsa__100813798m.jpg',
                'https://cdn.riastatic.com/photosnew/dom/photo/perevireno-prodaja-kvartira-vinnitsa-akademicheskiy-mikolayivska-ulitsa__100814247m.jpg',
                'https://cdn.riastatic.com/photosnew/dom/photo/perevireno-prodaja-kvartira-vinnitsa-akademicheskiy-mikolayivska-ulitsa__100814244m.jpg',
            ]
        ]
        for input_value, output_value in zip(input_values, output_values):
            self.assertEquals(parser.parse_images(input_value), output_value)


if __name__ == '__main__':
    unittest.main()