
import sqlite3
import os

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