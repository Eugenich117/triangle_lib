from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit

# Инициализация Spark сессии
spark = SparkSession.builder \
    .appName("ProductCategoryPairs") \
    .getOrCreate()

def get_product_category_pairs(products_df, categories_df, product_category_df):
    # Объединение датафреймов для получения пар "Продукт - Категория"
    product_category_joined = (product_category_df
        .join(products_df, "product_id", "left")
        .join(categories_df, "category_id", "left")
        .select(
            col("product_name"),
            col("category_name")
        )
    )

    # Получение продуктов без категорий
    products_without_categories = (products_df
        .join(product_category_df, "product_id", "left_anti")
        .select(
            col("product_name"),
            lit(None).alias("category_name")
        )
    )

    # Объединение результатов
    result_df = product_category_joined.union(products_without_categories)

    return result_df

# Пример использования (можно адаптировать под ваши данные)
if __name__ == "__main__":
    # Пример данных
    data_products = [(1, "Product A"), (2, "Product B"), (3, "Product C")]
    data_categories = [(1, "Category X"), (2, "Category Y")]
    data_product_category = [(1, 1), (1, 2), (2, 1)]

    # Создание датафреймов
    products_df = spark.createDataFrame(data_products, ["product_id", "product_name"])
    categories_df = spark.createDataFrame(data_categories, ["category_id", "category_name"])
    product_category_df = spark.createDataFrame(data_product_category, ["product_id", "category_id"])

    # Вызов метода
    result = get_product_category_pairs(products_df, categories_df, product_category_df)

    # Вывод результата
    result.show()