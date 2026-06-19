
import sqlite3
import os
import random

import pandas as pd

"""
SQLite helpers for IMPC image metadata tables.

The module provides small utilities for connecting to the database, selecting
rows, sampling work batches, updating image metadata, and evolving table
schemas as preprocessing and QC steps add new columns.
"""

DTYPE_MAP = {
    "int64": "INTEGER",
    "float64": "REAL",
    "object": "TEXT"
}

def connect_db(sqlite_file):
    """
    Open a connection to an existing SQLite database file.

    Parameters
    ----------
    sqlite_file : str | os.PathLike
        Path to the SQLite database file.

    Returns
    -------
    sqlite3.Connection
        Active SQLite connection.

    Raises
    ------
    FileNotFoundError
        Raised when the database file does not exist.
    sqlite3.OperationalError
        Raised when SQLite cannot open the database.
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
    Delete rows whose OMERO ID occurs more than once in a table.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active SQLite connection.
    db_table : str
        Table to clean.
    logger : logging.Logger, optional
        Logger used for status and error reporting.

    Returns
    -------
    None
        The function deletes ambiguous rows in place and commits the change.

    Raises
    ------
    ValueError
        Raised when `db_table` fails validation.
    sqlite3.Error
        Raised when the delete operation fails.
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
    Select rows from a table, optionally filtered by center.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active SQLite connection.
    db_table : str
        Table to query.
    center : str | None, optional
        Optional center name used as a filter.
    logger : logging.Logger, optional
        Logger used for status and error reporting.

    Returns
    -------
    pandas.DataFrame
        Query result as a dataframe.

    Raises
    ------
    ValueError
        Raised when `db_table` fails validation.
    sqlite3.Error
        Raised when the query fails.
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
    Select rows using one column/value filter.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active SQLite connection.
    db_table : str
        Table to query.
    column : str
        Column used as the filter key.
    value : object
        Value to match. `None` produces an `IS NULL` predicate.
    logger : logging.Logger, optional
        Currently unused placeholder for interface consistency.

    Returns
    -------
    pandas.DataFrame
        Query result as a dataframe.
    """
    return select_rows_by_columns(conn, db_table, filters={column: value})  

def select_rows_by_columns(conn, db_table, filters: dict):
    """
    Select rows using multiple column/value filters.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active SQLite connection.
    db_table : str
        Table to query.
    filters : dict
        Mapping of column names to desired values. `None` values produce
        `IS NULL` predicates.

    Returns
    -------
    pandas.DataFrame
        Query result as a dataframe.

    Raises
    ------
    ValueError
        Raised when any requested filter column is absent from the table.
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
    Randomly sample rows matching a column/value condition.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active SQLite connection.
    db_table : str
        Table to query.
    num_rows : int
        Number of random rows to select.
    column : str
        Column used for filtering.
    value : object
        Required value in `column`.
    center : str | None, optional
        Optional additional center filter.
    logger : logging.Logger, optional
        Currently unused placeholder for interface consistency.

    Returns
    -------
    pandas.DataFrame
        Random sample of matching rows. The sample is built with SQL `OFFSET`
        selection and may be smaller than `num_rows` when few matches exist.
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
    Randomly sample rows whose image files were already downloaded.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active SQLite connection.
    db_table : str
        Table to query.
    num_rows : int
        Number of random rows to select.
    center : str | None, optional
        Optional additional center filter.

    Returns
    -------
    pandas.DataFrame
        Random sample of rows where `downloaded = 'yes'`.
    """
    return select_random_rows_by_column(conn, db_table, num_rows=num_rows, column='downloaded', value='yes', center=center)

def select_next_download_batch(conn, db_table, limit, status_column="downloaded", status_value="yes", order_by="center", logger=None):
    """
    Select the next batch of rows that still need work.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active SQLite connection.
    db_table : str
        Table to query.
    limit : int
        Maximum number of rows to return.
    status_column : str, optional
        Column whose completed value should be excluded.
    status_value : object, optional
        Completed value to exclude from the batch.
    order_by : str, optional
        Column used to order the batch.
    logger : logging.Logger, optional
        Logger used for status and error reporting.

    Returns
    -------
    pandas.DataFrame
        Rows matching the batch criteria.

    Raises
    ------
    ValueError
        Raised when identifiers fail validation or `limit` is invalid.
    sqlite3.Error
        Raised when the query fails.
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
    Save basic download metadata for one image row.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active SQLite connection.
    db_table : str
        Table to update.
    status : str
        Download status value, typically `'yes'` or `'no'`.
    type : str
        Image type stored in the `type` column.
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.
    omero_id : str | int
        OMERO image identifier.
    mouse_id : str
        Mouse identifier.
    center : str
        Center name associated with the image.
    logger : logging.Logger, optional
        Logger used for status and error reporting.

    Returns
    -------
    int
        Number of rows affected by the update.

    Raises
    ------
    ValueError
        Raised when identifiers are invalid or required values are empty.
    sqlite3.Error
        Raised when the update fails.
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
    Save a dataframe of image metadata into an existing table.

    Parameters
    ----------
    df : pandas.DataFrame
        Metadata rows to persist. The dataframe must contain the key column
        expected by `update_rows_from_dataframe`.
    conn : sqlite3.Connection
        Active SQLite connection.
    db_table : str
        Table to update.
    logger : logging.Logger, optional
        Logger used for schema and update reporting.

    Returns
    -------
    int
        Total number of rows affected by the update phase.

    Raises
    ------
    ValueError
        Raised when the dataframe is empty or `db_table` is invalid.
    sqlite3.Error
        Raised when schema changes or updates fail.

    Notes
    -----
    Missing columns are added to the table before row updates are executed.
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
    Update table rows from a dataframe keyed by one column.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active SQLite connection.
    df : pandas.DataFrame
        Dataframe containing the update values. It must include `key_column`.
    db_table : str, optional
        Table to update.
    key_column : str, optional
        Column used to match database rows.
    logger : logging.Logger, optional
        Logger used for update reporting.

    Returns
    -------
    int
        Total number of rows affected by the batch update.

    Raises
    ------
    ValueError
        Raised when validation fails or there are no non-key columns to update.
    sqlite3.Error
        Raised when the update fails.
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
    Update one column for rows matching a simple equality condition.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active SQLite connection.
    db_table : str
        Table to update.
    column : str
        Column to modify.
    value : object
        New value to assign.
    condition_column : str
        Column used in the `WHERE` clause.
    condition_value : object
        Required value in `condition_column`.

    Returns
    -------
    int
        Number of rows affected by the update.
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
    Record preprocessing status, dimensions, and method flags for one image.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active SQLite connection.
    db_table : str
        Table to update.
    status : str
        Preprocessing status, typically `'yes'` or `'no'`.
    omero_id : str | int
        OMERO image identifier.
    preproc_methods : sequence[str]
        Method names that should be mirrored into `preproc_*` flag columns.
    img_height : int
        Height of the preprocessed image in pixels.
    img_width : int
        Width of the preprocessed image in pixels.
    logger : logging.Logger, optional
        Logger used for schema changes.

    Returns
    -------
    None
        The function updates the table in place and commits the changes.

    Notes
    -----
    Missing status, dimension, and per-method columns are added automatically.
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

def update_rotate_status(conn, db_table, status, omero_id, img_height, img_width, logger=None):
    """
    Record rotation status and output dimensions for one image.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active SQLite connection.
    db_table : str
        Table to update.
    status : str
        Rotation status value.
    omero_id : str | int
        OMERO image identifier.
    img_height : int
        Height of the rotated image in pixels.
    img_width : int
        Width of the rotated image in pixels.
    logger : logging.Logger, optional
        Logger used for schema changes.

    Returns
    -------
    None
        The function updates the table in place and commits the changes.

    Notes
    -----
    Missing rotation status and dimension columns are added automatically.
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
    Add a column to a table if it is not already present.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active SQLite connection.
    db_table : str
        Table to modify.
    column_name : str
        Name of the column to add.
    column_type : str
        SQLite column type such as `TEXT`, `INTEGER`, or `REAL`.
    default_value : object, optional
        Optional SQL default expression used when creating the column.
    logger : logging.Logger, optional
        Logger used for schema-change reporting.

    Returns
    -------
    None
        The function updates the schema in place and commits the change.
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
    Retrieve one column value for a given OMERO ID.

    Parameters
    ----------
    column : str
        Column to retrieve.
    omero_id : str | int
        OMERO image identifier to look up.
    conn : sqlite3.Connection
        Active SQLite connection.
    db_table : str
        Table to query.

    Returns
    -------
    object | None
        Column value for the matching OMERO ID, or `None` when no row matches.
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
