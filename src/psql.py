import os
import pandas as pd

from typing import Optional
from dotenv import load_dotenv
from sqlalchemy import (
    create_engine,
    text,
    MetaData,
    Table,
    Column,
    String,
    Float,
    Integer,
    DateTime,
    UniqueConstraint,
)
from typing import Optional, List, Dict, Any


class PostgresDB:
    def __init__(self):
        load_dotenv()

        self.connection_string = os.getenv("DATABASE_URL")

        if self.connection_string is None:
            raise ValueError(
                "Connection string must be provided or set in the environment variable DATABASE_URL"
            )

        self.engine = create_engine(self.connection_string)

    def create_table(
        self,
        table_name: str,
        schema: Dict[str, Any],
        unique_columns: Optional[List[str]] = None,
    ) -> None:
        """
        Create table with specified schema and uniqueness constraints

        Args:
            table_name: Name of the table to create
            schema: Dictionary mapping column names to SQLAlchemy types
            unique_columns: List of columns that should be unique together

        >> db.create_table_with_constraints(
        >>     'stock_data',
        >>     {
        >>         'ticker': String(10),
        >>         'ts': DateTime(timezone=True),
        >>         'open': Float,
        >>         'high': Float,
        >>         'low': Float,
        >>         'close': Float,
        >>         'vwap': Float,
        >>         'transactions': Integer
        >>     },
        >>     unique_columns=['ticker', 'ts']
        >> )
        """
        # Check if table already exists
        if self.engine.dialect.has_table(self.engine.connect(), table_name):
            print(f"Table '{table_name}' already exists.")
            return

        metadata = MetaData()

        # Build columns
        columns = []
        for col_name, col_type in schema.items():
            columns.append(Column(col_name, col_type))

        # Add unique constraint if specified
        constraints = []
        if unique_columns:
            constraints.append(
                UniqueConstraint(
                    *unique_columns, name=f"uq_{table_name}_{'_'.join(unique_columns)}"
                )
            )

        # Create table
        table = Table(table_name, metadata, *columns, *constraints)

        # Execute table creation
        with self.engine.connect() as conn:
            metadata.create_all(conn, tables=[table], checkfirst=True)
            conn.commit()

        print(
            f"Table '{table_name}' created with unique constraint on {unique_columns}"
        )

    def query(self, sql: str, params: Optional[dict] = None) -> pd.DataFrame:
        """
        Execute a SELECT query and return results as DataFrame

        >> df = db.query("SELECT * FROM my_table WHERE id = %(id)s", params={"id": 1})
        """
        return pd.read_sql(sql, self.engine, params=params)

    def execute(self, sql: str, params: Optional[dict] = None) -> None:
        """
        Execute INSERT/UPDATE/DELETE statements

        >> db.execute("INSERT INTO my_table (col1, col2) VALUES (%(val1)s, %(val2)s)",
        """
        with self.engine.connect() as conn:
            conn.execute(text(sql), params or {})
            conn.commit()

    def insert_df(self, df: pd.DataFrame, table_name: str, if_exists: str = "append"):
        """
        Insert DataFrame directly to table

        >> db.insert_df(df, 'my_table')
        """
        df.to_sql(table_name, self.engine, if_exists=if_exists, index=False)

    def table_to_df(self, table_name: str) -> pd.DataFrame:
        """
        Load entire table as DataFrame

        >> df = db.table_to_df('my_table')
        """
        return pd.read_sql_table(table_name, self.engine)
