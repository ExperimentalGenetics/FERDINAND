
import sqlite3
import os
import random

import pandas as pd

DTYPE_MAP = {
    "int64": "INTEGER",
    "float64": "REAL",
    "object": "TEXT"
}

def connect_db(sqlite_file):
    """
    Establishes a connection to the SQLite database and returns the connection object.
    Returns:
        sqlite3.Connection: A connection object to the SQLite database.
    Raises:
        sqlite3.OperationalError: If the database connection fails due to an invalid
        path or unavailable database file.
    """
    if os.path.exists(sqlite_file):
        try:
            connection = sqlite3.connect(sqlite_file)
            return connection
        except sqlite3.OperationalError as e:
            raise sqlite3.OperationalError(f"Failed to connect to database at {sqlite_file}: {e}")
    else:
        raise FileNotFoundError(f"ERROR: expected SQLite database {sqlite_file} not found!")
    
def remove_files_with_ambiguous_omero_ids(conn, db_table, logger=None):
    """
    Remove database records with ambiguous OMERO IDs (same ID linked to multiple mouse IDs).
    
    :param conn: active sqlite3 connection to database
    :param db_table: name of the table to clean
    :param logger: optional logger instance for logging results
    """
    
    # Validate table name to prevent SQL injection
    if not db_table.replace('_', '').isalnum():
        raise ValueError(f"Invalid table name: {db_table}")
    
    delete_query = f"DELETE FROM {db_table} WHERE omero_id IN (SELECT omero_id FROM {db_table} GROUP BY omero_id HAVING COUNT(*) > 1);"
    
    try:
        cursor = conn.cursor()
        cursor.execute(delete_query)
        deleted_rows = cursor.rowcount
        
        if logger is not None:
            logger.info(f"{deleted_rows} records deleted from table {db_table} due to ambiguous OMERO IDs")
        
        conn.commit()
    except sqlite3.Error as e:
        conn.rollback()
        if logger is not None:
            logger.error(f"Error deleting records from {db_table}: {e}")
        raise
    
def select_rows(conn, db_table, center=None, logger=None):
    """
    Retrieve all image files from the database, optionally filtered by center.
    
    :param conn: active sqlite3 connection to database
    :param db_table: name of the table to query
    :param center: optional center name to filter by
    :param logger: optional logger instance for logging results
    :return: pandas.DataFrame with the queried records
    """
    
    # Validate table name to prevent SQL injection
    if not db_table.replace('_', '').isalnum():
        raise ValueError(f"Invalid table name: {db_table}")
    
    try:
        if center is not None:
            query = f"SELECT * FROM {db_table} WHERE center = ?"
            df_result = pd.read_sql_query(query, conn, params=(center,))
            if logger is not None:
                logger.info(f"Retrieved {len(df_result)} records from {db_table} for center '{center}'")
        else:
            query = f"SELECT * FROM {db_table}"
            df_result = pd.read_sql_query(query, conn)
            if logger is not None:
                logger.info(f"Retrieved {len(df_result)} records from {db_table}")
        
        return df_result
    
    except sqlite3.Error as e:
        if logger is not None:
            logger.error(f"Error retrieving records from {db_table}: {e}")
        raise

def select_rows_by_column(conn, db_table, column, value, logger=None):
    """
    Query a SQL table for rows where the given column matches a value, or is NULL if value is None.
    
    :param conn: active sqlite3 connection to database
    :param db_table: name of the table to query
    :param column: name of the column to filter by (validated against table schema)
    :param value: value to match in the specified column (None for IS NULL check)
    :param logger: optional logger instance for logging results
    :return: pandas.DataFrame with the queried records
    """
    
    # Validate table name
    if not db_table.replace('_', '').isalnum():
        raise ValueError(f"Invalid table name: {db_table}")
    
    # Validate column name - must be alphanumeric or underscore
    if not column.replace('_', '').isalnum():
        raise ValueError(f"Invalid column name: {column}")
    
    # Verify column exists in table
    try:
        table_info = pd.read_sql_query(f"PRAGMA table_info({db_table})", conn)
        columns = table_info['name'].tolist()
        
        if column not in columns:
            raise ValueError(f"Column '{column}' does not exist in table '{db_table}'. "
                           f"Available columns: {columns}")
    except sqlite3.Error as e:
        if logger is not None:
            logger.error(f"Error validating column '{column}' in table '{db_table}': {e}")
        raise
    
    try:
        # Build and execute query
        if value is None:
            query = f"SELECT * FROM {db_table} WHERE {column} IS NULL"
            df_result = pd.read_sql_query(query, conn)
            if logger is not None:
                logger.info(f"Retrieved {len(df_result)} records from {db_table} where {column} IS NULL")
        else:
            query = f"SELECT * FROM {db_table} WHERE {column} = ?"
            df_result = pd.read_sql_query(query, conn, params=(value,))
            if logger is not None:
                logger.info(f"Retrieved {len(df_result)} records from {db_table} where {column} = '{value}'")
        
        return df_result
    
    except sqlite3.Error as e:
        if logger is not None:
            logger.error(f"Error querying {db_table} by column '{column}': {e}")
        raise

def select_random_rows_by_column(conn, db_table, num_rows: int, column, value, center=None, logger=None):
    """
    Select random rows from the database table where the specified column matches the given value, optionally filtered by center.
    
    :param conn: active sqlite3 connection to database
    :param db_table: name of the table to query
    :param num_rows: number of random rows to select
    :type num_rows: int
    :param column: name of the column to filter by (validated against table schema)
    :param value: value to match in the specified column
    :param center: name of the center (optional)
    :param logger: optional logger instance for logging results

    :return: pandas.DataFrame with the randomly selected records
    """
    cursor = conn.cursor()

    if center is not None:
        cursor.execute(f"SELECT COUNT(*) FROM {db_table} WHERE {column}=? AND center=?", (value, center))
        total_rows = cursor.fetchone()[0]
    else:
        cursor.execute(f"SELECT COUNT(*) FROM {db_table} WHERE {column}=?", (value,))
        total_rows = cursor.fetchone()[0]

    selected_rows = []
    if total_rows > 0:
        offsets = random.sample(range(total_rows), num_rows)

        for off in offsets:
            if center is not None:
                cursor.execute(f"SELECT * FROM {db_table} WHERE {column}=? AND center=? LIMIT 1 OFFSET ?", (value, center, off))
            else:
                cursor.execute(f"SELECT * FROM {db_table} WHERE {column}=? LIMIT 1 OFFSET ?", (value, off))
            row = cursor.fetchone()
            if row:
                selected_rows.append(row)

    # get column names from cursor.description
    col_names = [desc[0] for desc in cursor.description] if cursor.description else []

    # build DataFrame
    df = pd.DataFrame(selected_rows, columns=col_names)
    return df

def select_random_rows_with_images(conn, db_table, num_rows: int, center=None):
    return select_random_rows_by_column(conn, db_table, num_rows=num_rows, column='downloaded', value='yes', center=center)

def select_next_download_batch(conn, db_table, limit, status_column="downloaded", status_value="yes", order_by="center", logger=None):
    """
    Retrieve the next batch of records for which image files have not yet been downloaded from IMPC.
    
    :param conn: active sqlite3 connection to database
    :param db_table: name of the table to query
    :param limit: maximum number of records to retrieve
    :param status_column: column name indicating download status (default: 'downloaded')
    :param status_value: value indicating completed downloads (default: 'yes')
    :param order_by: column to sort results by (default: 'center')
    :param logger: optional logger instance for logging results
    :return: pandas.DataFrame with the next batch of records for which image files have not yet been downloaded
    """
    
    # Validate table name
    if not db_table.replace('_', '').isalnum():
        raise ValueError(f"Invalid table name: {db_table}")
    
    # Validate column names
    if not status_column.replace('_', '').isalnum():
        raise ValueError(f"Invalid status_column name: {status_column}")
    
    if not order_by.replace('_', '').isalnum():
        raise ValueError(f"Invalid order_by column name: {order_by}")
    
    # Validate limit
    try:
        limit = int(limit)
        if limit <= 0:
            raise ValueError(f"Limit must be a positive integer, got {limit}")
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid limit value: {e}")
    
    try:
        query = f"SELECT * FROM {db_table} WHERE {status_column} != ? ORDER BY {order_by} LIMIT ?"
        df_result = pd.read_sql_query(query, conn, params=(status_value, limit))
        
        if logger is not None:
            logger.info(f"Retrieved {len(df_result)} records from {db_table} where {status_column} != '{status_value}'")
        
        return df_result
    
    except sqlite3.Error as e:
        if logger is not None:
            logger.error(f"Error retrieving download batch from {db_table}: {e}")
        raise

def save_image_metadata(conn, db_table, status, type, width, height, omero_id, mouse_id, center, logger=None):
    """
    Update the database table with image metadata. Executes an SQL UPDATE to record 
    the latest metadata for the given image (dimensions, type, download status).
    
    :param conn: active sqlite3 connection to database
    :param db_table: name of the table to update
    :param status: status of the image download or processing (e.g., 'yes', 'no', 'failed')
    :param type: type of the image (e.g., 'JPEG', 'DICOM')
    :param width: width of the image in pixels
    :param height: height of the image in pixels
    :param omero_id: OMERO image ID associated with the image
    :param mouse_id: identifier of the mouse associated with the image
    :param center: name of the phenotyping centre or source of the image
    :param logger: optional logger instance for logging results
    :return: int - number of rows affected by the update
    :raises ValueError: If table name validation fails
    :raises sqlite3.Error: If database update fails
    """
    
    # Validate table name
    if not db_table.replace('_', '').isalnum():
        raise ValueError(f"Invalid table name: {db_table}")
    
    # Validate required parameters
    if not omero_id or not mouse_id or not center:
        raise ValueError("omero_id, mouse_id, and center cannot be empty")
    
    try:
        update_query = f"""
            UPDATE {db_table}
            SET downloaded = ?, type = ?, width = ?, height = ?
            WHERE omero_id = ? AND mouse_id = ? AND center = ?
        """
        
        params = (status, type, width, height, omero_id, mouse_id, center)
        cursor = conn.cursor()
        cursor.execute(update_query, params)
        affected_rows = cursor.rowcount
        
        conn.commit()
        
        if logger is not None:
            logger.info(f"Updated {affected_rows} record(s) for omero_id {omero_id}: "
                       f"status={status}, type={type}, dimensions={width}x{height}")
        
        return affected_rows
    
    except sqlite3.Error as e:
        conn.rollback()
        if logger is not None:
            logger.error(f"Error updating metadata in {db_table} for omero_id {omero_id}: {e}")
        raise

def save_full_image_metadata(df: pd.DataFrame, conn, db_table: str, logger=None):
    """
    Save image metadata to SQLite table, adding missing columns dynamically.
    
    :param df: pandas DataFrame containing image metadata to save
    :param conn: active sqlite3 connection to database
    :param db_table: name of the SQLite table to update/add columns to
    :param logger: optional logger instance for logging results
    :return: int - total number of rows affected by the update
    :raises ValueError: If table name validation fails or DataFrame is empty
    :raises sqlite3.Error: If database operations fail
    """
    
    # Validate inputs
    if df.empty:
        raise ValueError("DataFrame cannot be empty")
    
    if not db_table.replace('_', '').isalnum():
        raise ValueError(f"Invalid table name: {db_table}")
    
    try:
        cursor = conn.cursor()
        
        # Get existing columns in table
        cursor.execute(f"PRAGMA table_info({db_table})")
        existing_cols = [col[1] for col in cursor.fetchall()]
        
        # Add missing columns
        columns_added = []
        for col in df.columns:
            if col not in existing_cols:
                sqlite_type = DTYPE_MAP.get(str(df[col].dtype), "TEXT")
                cursor.execute(f"ALTER TABLE {db_table} ADD COLUMN {col} {sqlite_type}")
                columns_added.append(col)
                
                if logger is not None:
                    logger.info(f"Added column '{col}' ({sqlite_type}) to table '{db_table}'")
        
        conn.commit()
        
        if logger is not None and columns_added:
            logger.info(f"Schema update complete: {len(columns_added)} column(s) added to {db_table}")
        
        # Update table with DataFrame data
        affected_rows = update_rows_from_dataframe(conn, df, db_table=db_table, logger=logger)
        
        return affected_rows
    
    except sqlite3.Error as e:
        conn.rollback()
        if logger is not None:
            logger.error(f"Error saving image metadata to {db_table}: {e}")
        raise

def update_rows_from_dataframe(conn, df, db_table='impc_data', key_column='omero_id', logger=None):
    """
    Update SQLite table rows from a pandas DataFrame using the key_column as the unique identifier.
    
    :param conn: active sqlite3 connection to database
    :param df: pandas DataFrame with data to update
    :param db_table: name of the SQLite table to update (default: 'impc_data')
    :param key_column: column name used as unique key for update (default: 'omero_id')
    :param logger: optional logger instance for logging results
    :return: int - total number of rows affected by the update
    :raises ValueError: If table name, column name validation fails, or DataFrame is empty
    :raises sqlite3.Error: If database update fails
    """
    
    # Validate inputs
    if df.empty:
        raise ValueError("DataFrame cannot be empty")
    
    # Validate table name
    if not db_table.replace('_', '').isalnum():
        raise ValueError(f"Invalid table name: {db_table}")
    
    # Validate key column name
    if not key_column.replace('_', '').isalnum():
        raise ValueError(f"Invalid key_column name: {key_column}")
    
    # Check that key column exists in DataFrame
    if key_column not in df.columns:
        raise ValueError(f"Key column '{key_column}' not found in DataFrame columns: {df.columns.tolist()}")
    
    try:
        cursor = conn.cursor()
        
        # Get columns to update (all except key_column)
        update_cols = [col for col in df.columns if col != key_column]
        
        if not update_cols:
            raise ValueError(f"No columns to update (DataFrame only contains key_column '{key_column}')")
        
        # Build parameterized UPDATE query
        set_clause = ", ".join([f"{col} = ?" for col in update_cols])
        sql = f"UPDATE {db_table} SET {set_clause} WHERE {key_column} = ?"
        
        # Prepare values: update columns + key column
        values_list = []
        for _, row in df.iterrows():
            values = [row[col] for col in update_cols] + [row[key_column]]
            values_list.append(values)
        
        # Execute updates
        cursor.executemany(sql, values_list)
        affected_rows = cursor.rowcount
        
        conn.commit()
        
        if logger is not None:
            logger.info(f"Updated {affected_rows} row(s) in {db_table} using key column '{key_column}'")
        
        return affected_rows
    
    except sqlite3.Error as e:
        conn.rollback()
        if logger is not None:
            logger.error(f"Error updating {db_table} from DataFrame: {e}")
        raise

def update_column_values(conn, db_table, column, value, condition_column, condition_value):
    """
    Update a specific column in the database table based on a condition. This function executes an SQL UPDATE on the table
    to set the specified column to the given value where the condition is met.
    :param conn: active sqlite3 connection to database
    :param column: name of the column to update
    :param value: new value to set for the specified column
    :param condition_column: name of the column to use in the WHERE clause
    :param condition_value: value to match in the condition column
    :param table_name: name of the table to update
    """
    update = f"""
             UPDATE {db_table}
             SET    {column}=?
             WHERE  {condition_column}=?
             """

    params = (value, condition_value)
    cursor = conn.cursor()
    cursor.execute(update, params)
    conn.commit()

def update_preprocess_status(conn, db_table, status, omero_id, preproc_methods, img_height, img_width,
                             logger=None):
    """
    Update the database table with information about an image preprocessing status. This function executes an SQL UPDATE on the table
    to record the latest preprocessing status for the given image.
    :param conn: active sqlite3 connection to database
    :param status: status of the image preprocessing
    :param omero_id: OMERO image ID associated with the image
    :param preproc_methods: preprocessing methods applied to the image
    """
    add_column_to_table(conn, db_table, 'preprocessed', "TEXT", logger=logger)
    update = f"""
             UPDATE {db_table}
             SET    preprocessed=?
             WHERE  omero_id=? \
             """
    params = (status, omero_id)
    cursor = conn.cursor()
    cursor.execute(update, params)
    conn.commit()
    
    img_dims = {'preproc_height': img_height, 'preproc_width': img_width}
    for col, value in img_dims.items():
        add_column_to_table(conn, db_table, col, "INTEGER", logger=logger)
        update_dims = f"""
                         UPDATE {db_table}
                         SET    {col}=?
                         WHERE  omero_id=? \
                         """
        params = (value, omero_id,)
        cursor.execute(update_dims, params)
        conn.commit()

    for method in preproc_methods:
        method = f"preproc_{method}"
        add_column_to_table(conn, db_table, method, "TEXT", default_value="'no'", logger=logger)
        if status == 'yes':
            update_method = f"""
                             UPDATE {db_table}
                             SET    {method}='yes'
                             WHERE  omero_id=? \
                             """
            params = (omero_id,)
            cursor.execute(update_method, params)
            conn.commit()

def add_column_to_table(conn, db_table, column_name, column_type, default_value=None, logger=None):
    """
    Add a new column to an existing SQLite table if it does not already exist.
    :param conn: active sqlite3 connection to database
    :param table_name: name of the table to modify
    :param column_name: name of the new column to add
    :param column_type: data type of the new column (e.g., 'TEXT', 'INTEGER', 'REAL')
    :param default_value: optional default value for the new column
    """    
    cursor = conn.cursor()

    cursor.execute(f"PRAGMA table_info({db_table})")
    existing_cols = [col[1] for col in cursor.fetchall()]

    if column_name not in existing_cols:
        if default_value is not None:
            cursor.execute(f"ALTER TABLE {db_table} ADD COLUMN {column_name} {column_type} DEFAULT {default_value}")
        else:
            cursor.execute(f"ALTER TABLE {db_table} ADD COLUMN {column_name} {column_type}")
        if logger is not None: 
            logger.info(f" Added column '{column_name}' ({column_type}) to table '{db_table}'")

    conn.commit()