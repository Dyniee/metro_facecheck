CREATE DATABASE IF NOT EXISTS metro_facecheck CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE metro_facecheck;

CREATE TABLE IF NOT EXISTS users (
  id INT AUTO_INCREMENT PRIMARY KEY,
  username VARCHAR(100) UNIQUE NOT NULL,
  phone VARCHAR(20),
  email VARCHAR(150),
  password VARCHAR(255) NOT NULL,
  role ENUM('user','admin') DEFAULT 'user',
  face_registered TINYINT(1) DEFAULT 0
);

CREATE TABLE IF NOT EXISTS tickets (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NOT NULL,
  ticket_type ENUM('monthly','single') NOT NULL,
  purchase_time DATETIME NOT NULL,
  valid_from DATE,
  valid_to DATE,
  single_from_station VARCHAR(100),
  single_to_station VARCHAR(100),
  used TINYINT(1) DEFAULT 0,
  trip_code VARCHAR(100) UNIQUE,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS checkins (
  id INT AUTO_INCREMENT PRIMARY KEY,
  ticket_id INT,
  user_id INT,
  station VARCHAR(100),
  checkin_time DATETIME,
  success TINYINT(1),
  FOREIGN KEY (ticket_id) REFERENCES tickets(id) ON DELETE SET NULL,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
