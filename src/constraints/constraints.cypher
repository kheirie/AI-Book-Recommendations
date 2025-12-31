// Book: unique by bookID
CREATE CONSTRAINT book_id_unique IF NOT EXISTS
FOR (b:Book)
REQUIRE b.bookID IS UNIQUE;

// Author: unique by name (first_author or full name)
CREATE CONSTRAINT author_name_unique IF NOT EXISTS
FOR (a:Author)
REQUIRE a.name IS UNIQUE;

// Publisher: unique by name
CREATE CONSTRAINT publisher_name_unique IF NOT EXISTS
FOR (p:Publisher)
REQUIRE p.name IS UNIQUE;

// Genre: unique by genre name
CREATE CONSTRAINT genre_name_unique IF NOT EXISTS
FOR (g:Genre)
REQUIRE g.name IS UNIQUE;

// Category: unique by category name
CREATE CONSTRAINT category_name_unique IF NOT EXISTS
FOR (c:Category)
REQUIRE c.name IS UNIQUE;
