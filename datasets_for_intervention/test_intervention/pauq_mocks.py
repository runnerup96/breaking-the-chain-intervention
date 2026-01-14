class PAUQDatasetMock:
    def __init__(self):
        self.data = [
            {
                "index": 3,
                "query": "SELECT name ,  country ,  age FROM singer ORDER BY age DESC;",
                "question": "What are the names, countries, and ages for every singer in descending order of age?",
                "db": {'column_names': [[-1, '*'], [0, 'stadium id'], [0, 'location'], [0, 'name'], [0, 'capacity'], [0, 'highest'], [0, 'lowest'], [0, 'average'], [1, 'singer id'], [1, 'name'], [1, 'country'], [1, 'song name'], [1, 'song release year'], [1, 'age'], [1, 'is male'], [2, 'concert id'], [2, 'concert name'], [2, 'theme'], [2, 'stadium id'], [2, 'year'], [3, 'concert id'], [3, 'singer id']], 'column_names_original': [[-1, '*'], [0, 'Stadium_ID'], [0, 'Location'], [0, 'Name'], [0, 'Capacity'], [0, 'Highest'], [0, 'Lowest'], [0, 'Average'], [1, 'Singer_ID'], [1, 'Name'], [1, 'Country'], [1, 'Song_Name'], [1, 'Song_release_year'], [1, 'Age'], [1, 'Is_male'], [2, 'concert_ID'], [2, 'concert_Name'], [2, 'Theme'], [2, 'Stadium_ID'], [2, 'Year'], [3, 'concert_ID'], [3, 'Singer_ID']], 'column_types': ['text', 'number', 'text', 'text', 'number', 'number', 'number', 'number', 'number', 'text', 'text', 'text', 'text', 'number', 'others', 'number', 'text', 'text', 'text', 'text', 'number', 'text'], 'foreign_keys': [[18, 1], [21, 8], [20, 15]], 'primary_keys': [1, 8, 15, 20], 'table_names': ['stadium', 'singer', 'concert', 'singer in concert'], 'table_names_original': ['stadium', 'singer', 'concert', 'singer_in_concert']},
                "db_schema": {'stadium': ['stadium_id', 'location', 'name', 'capacity', 'highest', 'lowest', 'average'], 'singer': ['singer_id', 'name', 'country', 'song_name', 'song_release_year', 'age', 'is_male'], 'concert': ['concert_id', 'concert_name', 'theme', 'stadium_id', 'year'], 'singer_in_concert': ['concert_id', 'singer_id']},
                "true_schema_links": {'singer': ['age', 'country', 'name']},
                "paraphrase": "What are the singer names, residence countries, and ages, ordered in descending order based on age.",
                "schema_links": {'singer': ['age', 'country', 'name']},
                "generated_output": (
                    "===SCHEMA_LINKS===\n"
                    "singer:age\n"
                    "===SQL===\n"
                    "SELECT age, name FROM singer;\n\n"
                ),
                "generated_output_gold_structure": "SELECT age, name FROM singer;\n\n",
            },
            {
                "index": 23,
                "query": "SELECT T2.name ,  count(*) FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id  =  T2.stadium_id GROUP BY T1.stadium_id;",
                "question": "For each stadium, how many concerts play there?",
                "db": {'column_names': [[-1, '*'], [0, 'stadium id'], [0, 'location'], [0, 'name'], [0, 'capacity'], [0, 'highest'], [0, 'lowest'], [0, 'average'], [1, 'singer id'], [1, 'name'], [1, 'country'], [1, 'song name'], [1, 'song release year'], [1, 'age'], [1, 'is male'], [2, 'concert id'], [2, 'concert name'], [2, 'theme'], [2, 'stadium id'], [2, 'year'], [3, 'concert id'], [3, 'singer id']], 'column_names_original': [[-1, '*'], [0, 'Stadium_ID'], [0, 'Location'], [0, 'Name'], [0, 'Capacity'], [0, 'Highest'], [0, 'Lowest'], [0, 'Average'], [1, 'Singer_ID'], [1, 'Name'], [1, 'Country'], [1, 'Song_Name'], [1, 'Song_release_year'], [1, 'Age'], [1, 'Is_male'], [2, 'concert_ID'], [2, 'concert_Name'], [2, 'Theme'], [2, 'Stadium_ID'], [2, 'Year'], [3, 'concert_ID'], [3, 'Singer_ID']], 'column_types': ['text', 'number', 'text', 'text', 'number', 'number', 'number', 'number', 'number', 'text', 'text', 'text', 'text', 'number', 'others', 'number', 'text', 'text', 'text', 'text', 'number', 'text'], 'foreign_keys': [[18, 1], [21, 8], [20, 15]], 'primary_keys': [1, 8, 15, 20], 'table_names': ['stadium', 'singer', 'concert', 'singer in concert'], 'table_names_original': ['stadium', 'singer', 'concert', 'singer_in_concert']},
                "db_schema": {'stadium': ['stadium_id', 'location', 'name', 'capacity', 'highest', 'lowest', 'average'], 'singer': ['singer_id', 'name', 'country', 'song_name', 'song_release_year', 'age', 'is_male'], 'concert': ['concert_id', 'concert_name', 'theme', 'stadium_id', 'year'], 'singer_in_concert': ['concert_id', 'singer_id']},
                "true_schema_links": {'stadium': ['name']},
                "paraphrase": "For each stadium, what is the total count of concerts hosted?",
                "schema_links": {'stadium': ['name']},
                "generated_output": (
                    "===SCHEMA_LINKS===\n"
                    "stadium:name\n"
                    "===SQL===\n"
                    "SELECT name FROM stadium;\n\n"
                ),
            },
            {
                "index": 43,
                "query": "select count(*) from concert where stadium_id = (select stadium_id from stadium order by capacity desc limit 1);",
                "question": "Find the number of concerts happened in the stadium with the highest capacity .",
                "db": {'column_names': [[-1, '*'], [0, 'stadium id'], [0, 'location'], [0, 'name'], [0, 'capacity'], [0, 'highest'], [0, 'lowest'], [0, 'average'], [1, 'singer id'], [1, 'name'], [1, 'country'], [1, 'song name'], [1, 'song release year'], [1, 'age'], [1, 'is male'], [2, 'concert id'], [2, 'concert name'], [2, 'theme'], [2, 'stadium id'], [2, 'year'], [3, 'concert id'], [3, 'singer id']], 'column_names_original': [[-1, '*'], [0, 'Stadium_ID'], [0, 'Location'], [0, 'Name'], [0, 'Capacity'], [0, 'Highest'], [0, 'Lowest'], [0, 'Average'], [1, 'Singer_ID'], [1, 'Name'], [1, 'Country'], [1, 'Song_Name'], [1, 'Song_release_year'], [1, 'Age'], [1, 'Is_male'], [2, 'concert_ID'], [2, 'concert_Name'], [2, 'Theme'], [2, 'Stadium_ID'], [2, 'Year'], [3, 'concert_ID'], [3, 'Singer_ID']], 'column_types': ['text', 'number', 'text', 'text', 'number', 'number', 'number', 'number', 'number', 'text', 'text', 'text', 'text', 'number', 'others', 'number', 'text', 'text', 'text', 'text', 'number', 'text'], 'foreign_keys': [[18, 1], [21, 8], [20, 15]], 'primary_keys': [1, 8, 15, 20], 'table_names': ['stadium', 'singer', 'concert', 'singer in concert'], 'table_names_original': ['stadium', 'singer', 'concert', 'singer_in_concert']},
                "db_schema": {'stadium': ['stadium_id', 'location', 'name', 'capacity', 'highest', 'lowest', 'average'], 'singer': ['singer_id', 'name', 'country', 'song_name', 'song_release_year', 'age', 'is_male'], 'concert': ['concert_id', 'concert_name', 'theme', 'stadium_id', 'year'], 'singer_in_concert': ['concert_id', 'singer_id']},
                "true_schema_links": {'stadium': ['capacity']},
                "paraphrase": "What is the count of concerts that have taken place at the stadium with the largest seating capacity?",
                "schema_links": {'stadium': ['capacity']},
                "generated_output": (
                    "===SCHEMA_LINKS===\n"
                    "stadium:name\n"
                    "===SQL===\n"
                    "SELECT name FROM stadium;\n\n"
                ),
            },
            {
                "index": 177,
                "query": "SELECT T1.countryId ,  T1.CountryName FROM Countries AS T1 JOIN CAR_MAKERS AS T2 ON T1.CountryId  =  T2.Country GROUP BY T1.countryId HAVING count(*)  >  3 UNION SELECT T1.countryId ,  T1.CountryName FROM Countries AS T1 JOIN CAR_MAKERS AS T2 ON T1.CountryId  =  T2.Country JOIN MODEL_LIST AS T3 ON T2.Id  =  T3.Maker WHERE T3.Model  =  'fiat';",
                "question": "What are the id and names of the countries which have more than 3 car makers or produce the 'fiat' model?",
                "db": {'column_names': [[-1, '*'], [0, 'cont id'], [0, 'continent'], [1, 'country id'], [1, 'country name'], [1, 'continent'], [2, 'id'], [2, 'maker'], [2, 'full name'], [2, 'country'], [3, 'model id'], [3, 'maker'], [3, 'model'], [4, 'make id'], [4, 'model'], [4, 'make'], [5, 'id'], [5, 'mpg'], [5, 'cylinders'], [5, 'edispl'], [5, 'horsepower'], [5, 'weight'], [5, 'accelerate'], [5, 'year']], 'column_names_original': [[-1, '*'], [0, 'ContId'], [0, 'Continent'], [1, 'CountryId'], [1, 'CountryName'], [1, 'Continent'], [2, 'Id'], [2, 'Maker'], [2, 'FullName'], [2, 'Country'], [3, 'ModelId'], [3, 'Maker'], [3, 'Model'], [4, 'MakeId'], [4, 'Model'], [4, 'Make'], [5, 'Id'], [5, 'MPG'], [5, 'Cylinders'], [5, 'Edispl'], [5, 'Horsepower'], [5, 'Weight'], [5, 'Accelerate'], [5, 'Year']], 'column_types': ['text', 'number', 'text', 'number', 'text', 'number', 'number', 'text', 'text', 'text', 'number', 'number', 'text', 'number', 'text', 'text', 'number', 'text', 'number', 'number', 'text', 'number', 'number', 'number'], 'foreign_keys': [[5, 1], [9, 3], [11, 6], [14, 12], [16, 13]], 'primary_keys': [1, 3, 6, 10, 13, 16], 'table_names': ['continents', 'countries', 'car makers', 'model list', 'car names', 'cars data'], 'table_names_original': ['continents', 'countries', 'car_makers', 'model_list', 'car_names', 'cars_data']},
                "db_schema": {'continents': ['contid', 'continent'], 'countries': ['countryid', 'countryname', 'continent'], 'car_makers': ['id', 'maker', 'fullname', 'country'], 'model_list': ['modelid', 'maker', 'model'], 'car_names': ['makeid', 'model', 'make'], 'cars_data': ['id', 'mpg', 'cylinders', 'edispl', 'horsepower', 'weight', 'accelerate', 'year']},
                "true_schema_links": {'countries': ['countryid', 'countryname']},
                "paraphrase": "What countries possess either more than three automotive manufacturers or have 'Fiat' as one of their manufactured models, and what are their respective IDs and names?",
                "schema_links": {'countries': ['countryid', 'countryname']},
                "generated_output": (
                    "===SCHEMA_LINKS===\n"
                    "table1:column1,column2\n"
                    "===SQL===\n"
                    "SELECT column1, column2 FROM table1;\n\n"
                ),
            },
            {
                "index": 203,
                "query": 'SELECT count(*) FROM FLIGHTS WHERE SourceAirport  =  "APG";',
                "question": "How many flights depart from 'APG'?",
                "db": {'column_names': [[-1, '*'], [0, 'airline id'], [0, 'airline name'], [0, 'abbreviation'], [0, 'country'], [1, 'city'], [1, 'airport code'], [1, 'airport name'], [1, 'country'], [1, 'country abbrev'], [2, 'airline'], [2, 'flight number'], [2, 'source airport'], [2, 'destination airport']], 'column_names_original': [[-1, '*'], [0, 'uid'], [0, 'Airline'], [0, 'Abbreviation'], [0, 'Country'], [1, 'City'], [1, 'AirportCode'], [1, 'AirportName'], [1, 'Country'], [1, 'CountryAbbrev'], [2, 'Airline'], [2, 'FlightNo'], [2, 'SourceAirport'], [2, 'DestAirport']], 'column_types': ['text', 'number', 'text', 'text', 'text', 'text', 'text', 'text', 'text', 'text', 'number', 'number', 'text', 'text'], 'foreign_keys': [[13, 6], [12, 6]], 'primary_keys': [1, 6, 10], 'table_names': ['airlines', 'airports', 'flights'], 'table_names_original': ['airlines', 'airports', 'flights']},
                "db_schema": {'airlines': ['uid', 'airline', 'abbreviation', 'country'], 'airports': ['city', 'airportcode', 'airportname', 'country', 'countryabbrev'], 'flights': ['airline', 'flightno', 'sourceairport', 'destairport']},
                "true_schema_links": {'flights': ['sourceairport']},
                "paraphrase": "What is the count of flights that originate from the airport code 'APG'?",
                "schema_links": {'countries': ['countryid', 'countryname']},
                "generated_output": (
                    "===SCHEMA_LINKS===\n"
                    "table1:column1,column2\n"
                    "===SQL===\n"
                    "SELECT column1, column2 FROM table1;\n\n"
                ),
                "generated_output_gold_structure": (
                    "SELECT column1, column2 FROM table1;\n\n"
                )
            },
        ]
        self.tables = {'stadium': ['stadium_id', 'location', 'name', 'capacity', 'highest', 'lowest', 'average'], 'singer': ['singer_id', 'name', 'country', 'song_name', 'song_release_year', 'age', 'is_male'], 'concert': ['concert_id', 'concert_name', 'theme', 'stadium_id', 'year'], 'singer_in_concert': ['concert_id', 'singer_id'], 'continents': ['contid', 'continent'], 'countries': ['countryid', 'countryname', 'continent'], 'car_makers': ['id', 'maker', 'fullname', 'country'], 'model_list': ['modelid', 'maker', 'model'], 'car_names': ['makeid', 'model', 'make'], 'cars_data': ['id', 'mpg', 'cylinders', 'edispl', 'horsepower', 'weight', 'accelerate', 'year'], 'airlines': ['uid', 'airline', 'abbreviation', 'country'], 'airports': ['city', 'airportcode', 'airportname', 'country', 'countryabbrev'], 'flights': ['airline', 'flightno', 'sourceairport', 'destairport']}

        self.dummy_tables = {
            "space_missions": ["id", "name", "launch_date", "country", "success", "budget_mlns"],
            "albums": ["id", "artist_id", "title", "year", "genre", "duration_sec"],
            "matches": ["id", "tournament_id", "team_a", "team_b", "score_a", "score_b", "match_date"],
            "recipes": ["id", "dish_name", "cuisine", "calories", "cook_time_min", "difficulty"],
            "air_quality": ["id", "city", "measure_date", "pm25", "pm10", "aqi"]
        }

    @staticmethod
    def get_table_columns(db, table_name, original: bool = False):
        table_key = "table_names_original" if original else "table_names"
        column_key = "column_names_original" if original else "column_names"
        table_db_idx = db[table_key].index(table_name)
        table_columns = []
        for i, col_name in db[column_key][1:]:
            if i == table_db_idx:
                table_columns.append(col_name.lower())
        return table_columns

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


if __name__ == "__main__":
    dataset = PAUQDatasetMock()
    for item in dataset:
        print("=" * 100)
        for k, v in item.items():
            if isinstance(v, str):
                print(f'"{k}": "{v}",')
            else:
                print(f'"{k}": {v},')