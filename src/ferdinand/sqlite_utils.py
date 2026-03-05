
import sqlite3
import os
import random

import pandas as pd

"""
This module provides utility functions for interacting with a SQLite database containing metadata about images downloaded from the IMPC.
"""

DTYPE_MAP = {
    "int64": "INTEGER",
    "float64": "REAL",
    "object": "TEXT"
}

def connect_db(sqlite_file):
    """
    Connect to the SQLite database at the specified file path. Raises an error if the file does not exist or connection fails.
    :param sqlite_file: path to the SQLite database file
    :return: sqlite3.Connection object if connection is successful
    :raises FileNotFoundError: If the specified SQLite file does not exist
    :raises sqlite3.OperationalError: If there is an error connecting to the database
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
    Remove records from the specified database table where the same OMERO ID appears more than once, indicating ambiguity in image metadata.
    :param conn: active sqlite3 connection to database
    :param db_table: name of the table to clean
    :param logger: optional logger instance for logging results
    """
    # validate table name to prevent SQL injection
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
    Select rows from the specified database table, optionally filtered by center. Returns a pandas DataFrame with the results.
    :param conn: active sqlite3 connection to database
    :param db_table: name of the table to query
    :param center: name of the center to filter by (optional)
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
    Select rows from a SQL table based on a single column-value pair. If the value is None, selects rows where the column IS NULL.
    :param conn: active sqlite3 connection to database
    :param db_table: name of the table to query
    :param column: name of the column to filter by (validated against table schema)
    :param value: value to match in the specified column (can be None for IS NULL check)
    :param logger: optional logger instance for logging results
    :return: pandas.DataFrame with the queried records
    """
    return select_rows_by_columns(conn, db_table, filters={column: value})  

def select_rows_by_columns(conn, db_table, filters: dict):
    """
    Select rows from a SQL table based on multiple column-value pairs. If a value is None, selects rows where the column IS NULL.
    :param conn: active sqlite3 connection to database
    :param db_table: name of the table to query
    :param filters: dictionary of column-value pairs to filter by (value can be None for IS NULL check)
    :type filters: dict
    :return: pandas.DataFrame with the queried records
    """
    # check which column exists
    table_info = pd.read_sql_query(f"PRAGMA table_info({db_table})", conn)
    valid_columns = set(table_info["name"].tolist())

    invalid_columns = [col for col in filters.keys() if col not in valid_columns]
    if invalid_columns:
        raise ValueError(
            f"Invalid column(s): {invalid_columns}. "
            f"Available columns are: {sorted(valid_columns)}"
        )
    
    # build WHERE clause
    conditions = []
    params = []

    for col, val in filters.items():
        if val is None:
            conditions.append(f"{col} IS NULL")
        else:
            conditions.append(f"{col} = ?")
            params.append(val)

    where_clause = " AND ".join(conditions) if conditions else "1=1"
    query = f"SELECT * FROM {db_table} WHERE {where_clause}"

    return pd.read_sql_query(query, conn, params=params)

def select_random_rows_by_column(conn, db_table, num_rows: int, column, value, center=None, logger=None):
    """
    Select a random sample of rows from the specified database table where the given column matches the specified value, optionally filtered by center. Returns a pandas DataFrame with the results.
    :param conn: active sqlite3 connection to database
    :param db_table: name of the table to query
    :param num_rows: number of random rows to select
    :param column: name of the column to filter by (validated against table schema)
    :param value: value to match in the specified column
    :param center: name of the center to filter by (optional)
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
    """
    Select a random sample of rows from the specified database table where the 'downloaded' column is 'yes', indicating that image files have been downloaded, optionally filtered by center. Returns a pandas DataFrame with the results.
    :param conn: active sqlite3 connection to database
    :param db_table: name of the table to query
    :param num_rows: number of random rows to select
    :param center: name of the center to filter by (optional)
    :return: pandas.DataFrame with the randomly selected records where images have been downloaded
    """
    return select_random_rows_by_column(conn, db_table, num_rows=num_rows, column='downloaded', value='yes', center=center)

def select_next_download_batch(conn, db_table, limit, status_column="downloaded", status_value="yes", order_by="center", logger=None):
    """
    Select the next batch of rows from the specified database table where the status_column does not equal the status_value, ordered by the specified column, and limited to the specified number of rows. Returns a pandas DataFrame with the results.
    :param conn: active sqlite3 connection to database
    :param db_table: name of the table to query
    :param limit: maximum number of rows to return (must be a positive integer)
    :param status_column: name of the column to check for the status value (default: "downloaded")
    :param status_value: value to exclude in the status column (default: "yes")
    :param order_by: name of the column to order the results by (default: "center")
    :param logger: optional logger instance for logging results
    :return: pandas.DataFrame with the selected records for the next download batch
    :raises ValueError: If table name, column name validation fails, or limit is not a positive integer
    :raises sqlite3.Error: If database query fails
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
    Update the database table with metadata information about a downloaded image, including its download status, type, dimensions, and associated identifiers. 
    This function executes an SQL UPDATE on the table to record the latest metadata for the given image.
    :param conn: active sqlite3 connection to database
    :param db_table: name of the table to update
    :param status: download status of the image (e.g., 'yes' or 'no')
    :param type: type of the image (e.g., 'brightfield', 'fluorescence')
    :param width: width of the image in pixels
    :param height: height of the image in pixels
    :param omero_id: OMERO image ID associated with the image
    :param mouse_id: mouse ID associated with the image
    :param center: name of the center that provided the image
    :param logger: optional logger instance for logging results
    :return: int - number of rows affected by the update
    :raises ValueError: If table name validation fails or required parameters are empty
    :raises sqlite3.Error: If database update fails
    """
    # validate table name
    if not db_table.replace('_', '').isalnum():
        raise ValueError(f"Invalid table name: {db_table}")
    
    # validate required parameters
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
    Save full image metadata from a pandas DataFrame to the specified SQLite database table. 
    This function checks for missing columns in the table and adds them if necessary, then updates the table with the data from the DataFrame using the 'omero_id' as the unique identifier for each record.
    :param df: pandas DataFrame containing the image metadata to save (must include 'omero_id' column)
    :param conn: active sqlite3 connection to database
    :param db_table: name of the SQLite table to update
    :param logger: optional logger instance for logging results
    :return: int - total number of rows affected by the update
    :raises ValueError: If table name validation fails, DataFrame is empty, or 'omero_id' column is missing from DataFrame
    :raises sqlite3.Error: If database update fails
    """
    # validate inputs
    if df.empty:
        raise ValueError("DataFrame cannot be empty")
    
    if not db_table.replace('_', '').isalnum():
        raise ValueError(f"Invalid table name: {db_table}")
    
    try:
        cursor = conn.cursor()
        
        # get existing columns in table
        cursor.execute(f"PRAGMA table_info({db_table})")
        existing_cols = [col[1] for col in cursor.fetchall()]
        
        # add missing columns
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
        
        # update table with DataFrame data
        affected_rows = update_rows_from_dataframe(conn, df, db_table=db_table, logger=logger)
        
        return affected_rows
    
    except sqlite3.Error as e:
        conn.rollback()
        if logger is not None:
            logger.error(f"Error saving image metadata to {db_table}: {e}")
        raise

def update_rows_from_dataframe(conn, df, db_table='impc_data', key_column='omero_id', logger=None):
    """
    Update rows in the specified database table using data from a pandas DataFrame. 
    The function uses the specified key column to match records in the database and updates all other columns with the corresponding values from the DataFrame.
    :param conn: active sqlite3 connection to database
    :param df: pandas DataFrame containing the data to update (must include key_column)
    :param db_table: name of the database table to update (default: 'impc_data')
    :param key_column: name of the column to use as the unique identifier for matching records (default: 'omero_id')
    :param logger: optional logger instance for logging results
    :return: int - total number of rows affected by the update
    :raises ValueError: If table name validation fails, DataFrame is empty, key_column is invalid, or key_column is missing from DataFrame
    :raises sqlite3.Error: If database update fails
    """
    # validate inputs
    if df.empty:
        raise ValueError("DataFrame cannot be empty")
    
    # validate table name
    if not db_table.replace('_', '').isalnum():
        raise ValueError(f"Invalid table name: {db_table}")
    
    # validate key column name
    if not key_column.replace('_', '').isalnum():
        raise ValueError(f"Invalid key_column name: {key_column}")
    
    # check that key column exists in DataFrame
    if key_column not in df.columns:
        raise ValueError(f"Key column '{key_column}' not found in DataFrame columns: {df.columns.tolist()}")
    
    try:
        cursor = conn.cursor()
        
        # get columns to update (all except key_column)
        update_cols = [col for col in df.columns if col != key_column]
        
        if not update_cols:
            raise ValueError(f"No columns to update (DataFrame only contains key_column '{key_column}')")
        
        # build parameterized UPDATE query
        set_clause = ", ".join([f"{col} = ?" for col in update_cols])
        sql = f"UPDATE {db_table} SET {set_clause} WHERE {key_column} = ?"
        
        # prepare values: update columns + key column
        values_list = []
        for _, row in df.iterrows():
            values = [row[col] for col in update_cols] + [row[key_column]]
            values_list.append(values)
        
        # execute updates
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
    Update a specific column in the database table based on a condition. 
    This function executes an SQL UPDATE on the table to set the specified column to the given value where the condition is met.
    :param conn: active sqlite3 connection to database
    :param db_table: name of the table to update
    :param column: name of the column to update
    :param value: new value to set in the specified column
    :param condition_column: name of the column to use in the WHERE clause for the condition
    :param condition_value: value to match in the condition column for the update to be applied
    :return: int - number of rows affected by the update
    :raises ValueError: If table name validation fails or column names are invalid
    :raises sqlite3.Error: If database update fails
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

    return cursor.rowcount  # Return the number of rows affected by the update

def update_preprocess_status(conn, db_table, status, omero_id, preproc_methods, img_height, img_width,
                             logger=None):
    """
    Update the database table with information about an image preprocessing status. 
    This function executes an SQL UPDATE on the table to record the latest preprocessing status for the given image, as well as the methods used and the dimensions
    of the preprocessed image.
    :param conn: active sqlite3 connection to database
    :param db_table: name of the table to update
    :param status: status of the image preprocessing (e.g., 'yes' or 'no')
    :param omero_id: OMERO image ID associated with the image
    :param preproc_methods: list of preprocessing methods applied to the image (e.g., ['denoise', 'normalize'])
    :param img_height: height of the preprocessed image in pixels
    :param img_width: width of the preprocessed image in pixels
    :param logger: optional logger instance for logging results"""
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

def update_rotate_status(conn, db_table, status, omero_id, img_height, img_width, logger=None):
    """
    Update the database table with information about an image rotation status. 
    This function executes an SQL UPDATE on the table to record the latest rotation status for the given image.
    :param conn: active sqlite3 connection to database
    :param db_table: name of the table to update
    :param status: status of the image rotation
    :param omero_id: OMERO image ID associated with the image
    :param img_height: height of the rotated image in pixels
    :param img_width: width of the rotated image in pixels
    :param logger: optional logger instance for logging results
    """
    colname = 'rotated'
    add_column_to_table(conn, db_table, colname, "TEXT", logger=logger)
    update = f"""
             UPDATE {db_table}
             SET    {colname}=?
             WHERE  omero_id=? \
             """
    params = (status, omero_id)
    cursor = conn.cursor()
    cursor.execute(update, params)
    conn.commit()
    
    img_dims = {'rotated_height': img_height, 'rotated_width': img_width}
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

def get_value_by_id(column, omero_id, conn, db_table):
    """
    Retrieve the value of a specific column from the database table for a given OMERO ID.
    :param column: name of the column to retrieve
    :param omero_id: OMERO image ID to look up
    :param conn: active sqlite3 connection to database
    :param db_table: name of the table to query
    :return: value of the specified column for the given OMERO ID, or None if not found
    """
    query = f"""
            SELECT {column} 
            FROM   {db_table}
            WHERE  omero_id=?
            """

    params = (omero_id,)
    cursor = conn.cursor()
    cursor.execute(query, params)
    result = cursor.fetchone()

    return result[0] if result else None