CREATE TABLE patients (
    patient_id SERIAL PRIMARY KEY,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    birth_date DATE,
    gender VARCHAR(10),
    contact_info TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE files (
    file_id SERIAL PRIMARY KEY,
    file_name VARCHAR(255) NOT NULL,
    bucket_name VARCHAR(255) NOT NULL,
    file_path VARCHAR(255) NOT NULL,
    file_type VARCHAR(50),  -- Можно расширить типы, если будут другие файлы
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    patient_id INT REFERENCES patients(patient_id),  -- Связь с таблицей пациентов
    image_type VARCHAR(50),  -- Тип изображения (например, dermoscopic, photograph)
    processed BOOLEAN DEFAULT FALSE,  -- Флаг, указывающий, обработано ли изображение
    -- classification_result_id INT REFERENCES classification_results(result_id),  -- Ссылка на результаты классификации
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE classification_results (
    result_id SERIAL PRIMARY KEY,
    image_id INT REFERENCES files(file_id),
    classification_type VARCHAR(50),
    probability FLOAT,
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE processed_data (
    processed_data_id SERIAL PRIMARY KEY,
    image_id INT REFERENCES files(file_id),
    preprocessing_type VARCHAR(50),
    processed_data_path VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE classification_models (
    model_id SERIAL PRIMARY KEY,
    model_name VARCHAR(100),
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE logs (
    log_id SERIAL PRIMARY KEY,
    action VARCHAR(100),
    status VARCHAR(50),
    message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    login TEXT UNIQUE NOT NULL,
    "firstName" TEXT NOT NULL,
    "lastName" TEXT NOT NULL,
    email TEXT UNIQUE,
    password TEXT NOT NULL,
    last_login TIMESTAMP WITH TIME ZONE,
    last_login_ip TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
