import unittest

import project.bot.utilities as utils


class UtilitiesTests(unittest.TestCase):
    def setUp(self) -> None:
        return None

    def tearDown(self) -> None:
        return None

    def test_find_function(self):
        input_values = [
            {
                'app_url': '',
                'models': [
                    {
                        'perms': {
                            'add': True,
                            'change': True,
                            'delete': True
                        },
                        'add_url': '/admin/cms/news/add/',
                        'admin_url': '/admin/cms/news/',
                        'name': ''
                    }
                ],
                'has_module_perms': True,
                'name': u'CMS'
            },
            {
                'app_url': '',
                'models': [
                    {
                        'perms': {
                            'add': True,
                            'change': True,
                            'delete': True
                        },
                        'add_url': '/admin/cms/news/add/',
                        'admin_url': '/admin/cms/news/',
                        'name': ''
                    }
                ],
                'has_module_perms': True,
                'name': u'CMS'
            },
            {
                'dataForFinalPage': {
                    'realty': {
                        'realty_type_parent_name_uk': 'Житло',
                        'street_name': 'Свердлова улица',
                        'rooms_count': 2,
                        'wall_type_uk': 'цегла',
                        'state_name': 'Винницкая',
                        'district_type_name': 'Район',
                        'description': 'Автономное отопление, счетчики .Гардеробная, окна во двор.Не угловая, не лифтовая!!!',
                        'currency_type': '$',
                        'longitude': '',
                        'inspected_at': '2019-06-13 13:06:45',
                        'is_exchange': 'нет',
                        'wall_type': 'кирпич',
                        'district_name_uk': 'Свердловський масив',
                        'publishing_date': '2019-08-12 11:27:13',
                        'description_uk': '',
                        'currency_type_uk': '$',
                        'price_type': 'за объект',
                        'price': 29999,
                        'user_newbuild_name': 'ЖК Ривьера',
                        'floor': 10,
                        'return_on_moderation': 2,
                        'moderation_date': '2019-08-06 17:15:26',
                        'latitude': '',
                        'price_total': 29999,
                        'is_exchange_uk': 'ні',
                        'realty_type_name': 'Квартира',
                        'building_number_str': '118',
                        'city_name': 'Винница',
                        'realty_type_parent_id': 1,
                        'rangeFactor': '00331565598433',
                        'living_square_meters': 40,
                        'state_name_uk': 'Вінницька',
                        'advert_type_name_uk': 'продаж',
                        'realty_type_id': 2,
                        'user_newbuild_name_uk': "ЖК Рів'єра",
                        'realty_type_name_uk': 'Квартира',
                        'floors_count': 12,
                        'created_at': '2019-06-06 10:37:54',
                        'kitchen_square_meters': 18,
                        'city_name_uk': 'Вінниця',
                        'flat_number': '',
                        'total_square_meters': 71.5,
                        'street_name_uk': 'Свердлова вулиця',
                        'district_name': 'Свердловский массив',
                        'realty_type_parent_name': 'Жилье',
                        'priceArr': {
                            '1': '29 999',
                            '2': '26 793',
                            '3': '763 333'
                        },
                        'mainCharacteristics': {
                            'chars': [
                                {
                                    'name': 'Комнат',
                                    'value': 2
                                },
                                {
                                    'name': 'Этаж',
                                    'value': 10
                                },
                                {
                                    'name': 'Этажность',
                                    'value': 12
                                },
                                {
                                    'name': 'Площадь',
                                    'value': [
                                        71.5,
                                        40,
                                        18
                                    ],
                                    'units': 'м²'
                                },
                                {
                                    'name': 'Тип предложения',
                                    'value': 'от посредника'
                                },
                                {
                                    'name': 'Тип стен',
                                    'value': 'кирпич'
                                },
                                {
                                    'name': 'Отопление',
                                    'value': 'без отопления'
                                }
                            ]
                        }
                    },
                    'agencyOwner': {
                        'owner': {
                            'name': 'Карина'
                        }
                    }
                }
            },
            {
                'dataForFinalPage': {
                    'realty': {
                        'priceArr': {
                            '1': '29 999',
                            '2': '26 793',
                            '3': '763 333'
                        },
                        'mainCharacteristics': {
                            'chars': [
                                {
                                    'name': 'Комнат',
                                    'value': 2
                                },
                                {
                                    'name': 'Этаж',
                                    'value': 10
                                },
                                {
                                    'name': 'Этажность',
                                    'value': 12
                                },
                                {
                                    'name': 'Площадь',
                                    'value': [
                                        71.5,
                                        40,
                                        18
                                    ],
                                    'units': 'м²'
                                },
                                {
                                    'name': 'Тип предложения',
                                    'value': 'от посредника'
                                },
                                {
                                    'name': 'Тип стен',
                                    'value': 'кирпич'
                                },
                                {
                                    'name': 'Отопление',
                                    'value': 'без отопления'
                                }
                            ]
                        }
                    },
                    'agencyOwner': {
                        'owner': {
                            'name': 'Карина'
                        }
                    }
                }
            },
            {
                'dataForFinalPage': {
                    'realty': {
                        'priceArr': {
                            '1': '29 999',
                            '2': '26 793',
                            '3': '763 333'
                        },
                        'mainCharacteristics': {
                            'chars': [
                                {
                                    'name': 'Комнат',
                                    'value': 2
                                },
                                {
                                    'name': 'Этаж',
                                    'value': 10
                                },
                                {
                                    'name': 'Этажность',
                                    'value': 12
                                },
                                {
                                    'name': 'Площадь',
                                    'value': [
                                        71.5,
                                        40,
                                        18
                                    ],
                                    'units': 'м²'
                                },
                                {
                                    'name': 'Тип предложения',
                                    'value': 'от посредника'
                                },
                                {
                                    'name': 'Тип стен',
                                    'value': 'кирпич'
                                },
                                {
                                    'name': 'Отопление',
                                    'value': 'без отопления'
                                }
                            ]
                        }
                    },
                    'agencyOwner': {
                        'owner': {
                            'name': 'Карина'
                        }
                    }
                }
            }
        ]
        output_values = [
            ['/admin/cms/news/'],
            [True],
            [
                {
                    '1': '29 999',
                    '2': '26 793',
                    '3': '763 333'
                }
            ],
            [
                [
                    {
                        'name': 'Комнат',
                        'value': 2
                    },
                    {
                        'name': 'Этаж',
                        'value': 10
                    },
                    {
                        'name': 'Этажность',
                        'value': 12
                    },
                    {
                        'name': 'Площадь',
                        'value': [
                            71.5,
                            40,
                            18
                        ],
                        'units': 'м²'
                    },
                    {
                        'name': 'Тип предложения',
                        'value': 'от посредника'
                    },
                    {
                        'name': 'Тип стен',
                        'value': 'кирпич'
                    },
                    {
                        'name': 'Отопление',
                        'value': 'без отопления'
                    }
                ]
            ],
            [
                'Комнат', 'Этаж', 'Этажность', 'Площадь', 'Тип предложения', 'Тип стен', 'Отопление', 'Карина'
            ]
        ]
        feature_values = [
            'admin_url',
            'change',
            'priceArr',
            'chars',
            'name'
        ]
        for input_value, output_value, feature_value in zip(input_values, output_values, feature_values):
            found_value = list(utils.find(feature_value, input_value))
            self.assertEquals(
                first=found_value,
                second=output_value
            )

    def test_find_feature_in_function(self):
        input_values = [
            {
                'dataForFinalPage': {
                    'realty': {
                        'priceArr': {
                            '1': '29 999',
                            '2': '26 793',
                            '3': '763 333'
                        },
                        'mainCharacteristics': {
                            'chars': [
                                {
                                    'name': 'Комнат',
                                    'value': 2
                                },
                                {
                                    'name': 'Этаж',
                                    'value': 10
                                },
                                {
                                    'name': 'Этажность',
                                    'value': 12
                                },
                                {
                                    'name': 'Площадь',
                                    'value': [
                                        71.5,
                                        40,
                                        18
                                    ],
                                    'units': 'м²'
                                },
                                {
                                    'name': 'Тип предложения',
                                    'value': 'от посредника'
                                },
                                {
                                    'name': 'Тип стен',
                                    'value': 'кирпич'
                                },
                                {
                                    'name': 'Отопление',
                                    'value': 'без отопления'
                                }
                            ]
                        }
                    },
                    'agencyOwner': {
                        'owner': {
                            'name': 'Карина'
                        }
                    }
                }
            },
            {
                'dataForFinalPage': {
                    'realty': {
                        'priceArr': {
                            '1': '29 999',
                            '2': '26 793',
                            '3': '763 333'
                        },
                        'mainCharacteristics': {
                            'chars': [
                                {
                                    'name': 'Комнат',
                                    'value': 2
                                },
                                {
                                    'name': 'Этаж',
                                    'value': 10
                                },
                                {
                                    'name': 'Этажность',
                                    'value': 12
                                },
                                {
                                    'name': 'Площадь',
                                    'value': [
                                        71.5,
                                        40,
                                        18
                                    ],
                                    'units': 'м²'
                                },
                                {
                                    'name': 'Тип предложения',
                                    'value': 'от посредника'
                                },
                                {
                                    'name': 'Тип стен',
                                    'value': 'кирпич'
                                },
                                {
                                    'name': 'Отопление',
                                    'value': 'без отопления'
                                }
                            ]
                        }
                    },
                    'agencyOwner': {
                        'owner': {
                            'name': 'Карина'
                        }
                    }
                }
            },
            {
                'dataForFinalPage': {
                    'realty': {
                        'priceArr': {
                            '1': '29 999',
                            '2': '26 793',
                            '3': '763 333'
                        },
                        'mainCharacteristics': {
                            'chars': [
                                {
                                    'name': 'Комнат',
                                    'value': 2
                                },
                                {
                                    'name': 'Этаж',
                                    'value': 10
                                },
                                {
                                    'name': 'Этажность',
                                    'value': 12
                                },
                                {
                                    'name': 'Площадь',
                                    'value': [
                                        71.5,
                                        40,
                                        18
                                    ],
                                    'units': 'м²'
                                },
                                {
                                    'name': 'Тип предложения',
                                    'value': 'от посредника'
                                },
                                {
                                    'name': 'Тип стен',
                                    'value': 'кирпич'
                                },
                                {
                                    'name': 'Отопление',
                                    'value': 'без отопления'
                                }
                            ]
                        }
                    },
                    'agencyOwner': {
                        'owner': {
                            'name': 'Карина'
                        }
                    }
                }
            }
        ]
        output_values = [
            'без отопления',
            [
                71.5,
                40,
                18
            ],
            None
        ]
        feature_values = [
            'Отопление',
            'Площадь',
            'Год постройки'
        ]
        for input_value, output_value, feature_value in zip(input_values, output_values, feature_values):
            generator_obj = utils.find(
                dictionary=input_value,
                feature='chars'
            )
            self.assertEquals(
                first=utils.find_feature_in(
                    generator_obj=generator_obj,
                    feature=feature_value
                ),
                second=output_value
            )