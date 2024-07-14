from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from pymongo import MongoClient
from bson import ObjectId

app = Flask(__name__, static_url_path='/static')
app.secret_key = 'your_secret_key'  # Set a secret key for session management

# MongoDB connection
client = MongoClient('localhost', 27017)
db = client.learn



# Function to get recommendations for all user ingredients
def get_recommendations(page_number=1, items_per_page=5):
    # Fetching recipes and user ingredients from MongoDB
    all_recipes = list(db.TestCol.find({}, {"RecipeName": 1, "Ingredients": 1}))
    recipes_df = pd.DataFrame(all_recipes)

    user_ingredients = list(db.Ingredients_db.find({}, {"name": 1}))
    user_ingredients_df = pd.DataFrame(user_ingredients)

    # TF-IDF vectorization
    recipes_df["Ingredients"] = recipes_df["Ingredients"].replace(np.nan, "", regex=True)
    user_ingredients_combined = ",".join(user_ingredients_df["name"])
    tfidf_vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix_recipes = tfidf_vectorizer.fit_transform(list(recipes_df["Ingredients"]))
    user_vector = tfidf_vectorizer.transform([user_ingredients_combined])

    # Recommendation logic
    cosine_similarities = linear_kernel(user_vector, tfidf_matrix_recipes).flatten()
    recipe_scores = list(zip(recipes_df["RecipeName"], cosine_similarities))
    recipe_scores.sort(key=lambda x: x[1], reverse=True)

    start_idx = (page_number - 1) * items_per_page
    end_idx = start_idx + items_per_page
    recommended_recipes = [(title, score) for title, score in recipe_scores[start_idx:end_idx] if score > 0]
    return recommended_recipes

# Function to get recommendations for a specific ingredient
def get_recommendations_for_ingredient(ingredient_name, items_per_page=20):
    # Fetching recipes for a specific ingredient from MongoDB
    recipes_for_ingredient = list(db.TestCol.find({"Ingredients": {"$regex": ingredient_name, "$options": "i"}},
                                                  {"RecipeName": 1, "Ingredients": 1, "image-url": 1}))
    recipes_df = pd.DataFrame(recipes_for_ingredient)

    if recipes_df.empty:
        return []  # Return an empty list if no recipes are found for the ingredient

    recipes_df = recipes_df.drop_duplicates(subset=['RecipeName'])

    # TF-IDF vectorization
    recipes_df["Ingredients"] = recipes_df["Ingredients"].replace(np.nan, "", regex=True)
    tfidf_vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix_recipes = tfidf_vectorizer.fit_transform(list(recipes_df["Ingredients"]))

    # Recommendation logic
    user_vector = tfidf_vectorizer.transform([ingredient_name])
    cosine_similarities = linear_kernel(user_vector, tfidf_matrix_recipes).flatten()
    recipe_scores = list(zip(recipes_df["RecipeName"], cosine_similarities, recipes_df["image-url"]))
    recipe_scores.sort(key=lambda x: x[1], reverse=True)

    recommended_recipes = [(title, score, image_url) for title, score, image_url in recipe_scores if score > 0]
    return recommended_recipes

def calculate_missing_ingredients_count(recipe_name, ingredient_name):
    # Query the database to get the list of ingredients for the recipe
    recipe_details = db.TestCol.find_one({"RecipeName": recipe_name}, {"Cleaned-Ingredients": 1})
    if not recipe_details:
        return -1  # Recipe not found

    # Get the list of cleaned ingredients for the recipe
    recipe_ingredients = recipe_details.get("Cleaned-Ingredients", "").split(',')

    # Fetch the list of user ingredients
    user_ingredients = list(db.Ingredients_db.find({}, {"name": 1}))
    user_ingredients_list = [ingredient["name"].lower() for ingredient in user_ingredients]

    # Initialize the missing ingredients count
    missing_ingredients_count = 0

    # Iterate over recipe ingredients
    for ingredient in recipe_ingredients:
        # Check if the ingredient is not in the user's inventory
        if ingredient.strip().lower() not in user_ingredients_list:
            # Increment the missing ingredients count
            missing_ingredients_count += 1

    return missing_ingredients_count

# Flask route to handle displaying recipes for a specific ingredient
@app.route('/recipes/<ingredient_name>')
def recipes_for_ingredient(ingredient_name):
    # Get recommendations for the selected ingredient
    recommended_recipes = get_recommendations_for_ingredient(ingredient_name)

    # Add pagination logic
    total_recipes = len(recommended_recipes)
    items_per_page = 15
    total_pages = (total_recipes + items_per_page - 1) // items_per_page

    # Ensure the current page is within a valid range
    page = request.args.get('page', 1, type=int)
    page = max(1, min(page, total_pages))

    start_idx = (page - 1) * items_per_page
    end_idx = start_idx + items_per_page

    # Slice the recipes based on pagination
    paginated_recipes = recommended_recipes[start_idx:end_idx]

    # Calculate missing ingredients count for each recipe
    recipes_with_count = []
    for recipe_name, score, image_url in paginated_recipes:
        missing_ingredients_count = calculate_missing_ingredients_count(recipe_name, ingredient_name)
        recipes_with_count.append((recipe_name, score, image_url, missing_ingredients_count))

    # Define the get_recipe_url function in the context
    def get_recipe_url(recipe_name):
        recipe_details = db.TestCol.find_one({"RecipeName": recipe_name})
        return recipe_details.get("URL", "")

    # Pass the get_recipe_url function to the template context
    return render_template('recipes_for_ingredient.html', recipes=recipes_with_count, ingredient_name=ingredient_name, total_pages=total_pages, current_page=page, get_recipe_url=get_recipe_url)




# Flask route to display missing ingredients list for a recipe
@app.route('/missing-ingredients/<recipe_name>')
def missing_ingredients_list(recipe_name):
    # Fetch the recipe details from the database
    recipe_details = db.TestCol.find_one({"RecipeName": recipe_name})
    if not recipe_details:
        # If the recipe is not found, render a template indicating the error
        return render_template('recipe_not_found.html', recipe_name=recipe_name)

    # Get the list of ingredients for the recipe
    recipe_ingredients = recipe_details.get("Cleaned-Ingredients", "").split(',')

    # Fetch all user ingredients from MongoDB
    user_ingredients = list(db.Ingredients_db.find({}, {"name": 1}))
    user_ingredients_list = [ingredient["name"].lower() for ingredient in user_ingredients]

    # Find the missing ingredients for the recipe
    missing_ingredients = [ingredient for ingredient in recipe_ingredients if ingredient.strip().lower() not in user_ingredients_list]

    # Render the template with the missing ingredient list
    return render_template('missing_ingredients.html', recipe_name=recipe_name, missing_ingredients=missing_ingredients)


# Function to add selected ingredients to the shopping list session variable
def add_to(selected_ingredients):
    # Retrieve the shopping list from the session or initialize it if it doesn't exist
    shopping_list = session.get('shopping_list', [])
    # Extend the shopping list with the selected ingredients
    shopping_list.extend(selected_ingredients)
    # Update the shopping list in the session
    session['shopping_list'] = shopping_list

# Flask route to handle adding selected ingredients to the shopping list
@app.route('/add-to-shopping-list/<recipe_name>', methods=['POST'])
def add_to_shopping_list_route(recipe_name):
    if request.method == 'POST':
        selected_ingredients = request.form.getlist('selected_ingredients')
        add_to(selected_ingredients)  # Add selected ingredients to shopping list session variable
        print("Selected ingredients added to shopping list:", selected_ingredients)  # Debugging print statement
        return redirect(url_for('shopping_list'))  # Redirect to the shopping list page
    else:
        # Handle GET requests to this route (if any)
        pass
# Flask route to render the template with recommended recipes
@app.route('/')
def index():
    # Fetch all user ingredients from MongoDB
    user_ingredients = list(db.Ingredients_db.find({}, {"name": 1}))
    ingredients_list = [ingredient["name"] for ingredient in user_ingredients]
    return render_template('index.html',
     ingredients=ingredients_list)

# Flask route to handle the recommendation button click
@app.route('/recommend', methods=['POST'])
def recommend():
    # Update recommendations when the button is clicked
    page_number = request.json.get('page', 1)
    recommended_recipes = get_recommendations(page_number=page_number)
    total_recipes = len(recommended_recipes)
    items_per_page = 15
    total_pages = (total_recipes + items_per_page - 1) // items_per_page

    response = {
        'recommendations': recommended_recipes,
        'totalPages': total_pages
    }

    return jsonify(response)

# Flask route to render the shopping list
@app.route('/shopping-list')
def shopping_list():
    # Retrieve the selected ingredients from the session
    shopping_list = session.get('shopping_list', [])
    return render_template('shopping_list.html', shopping_list=shopping_list)

# Flask route to handle deleting an ingredient from the shopping list
@app.route('/delete-from-shopping-list/<ingredient>', methods=['POST'])
def delete_from_shopping_list(ingredient):
    if request.method == 'POST':
        # Retrieve the shopping list from the session
        shopping_list = session.get('shopping_list', [])
        # Remove the ingredient from the shopping list if it exists
        if ingredient in shopping_list:
            shopping_list.remove(ingredient)
            session['shopping_list'] = shopping_list
            return jsonify({'success': True, 'message': 'Ingredient deleted successfully'})
        else:
            return jsonify({'success': False, 'message': 'Ingredient not found in the shopping list'})

# Function to update the user's ingredient list with selected ingredients
def update_user_ingredients(selected_ingredients):
    # Update the user's ingredient list in the database
    db.Ingredients_db.insert_one({"ingredients": selected_ingredients})
    # Function to add ingredient to the ingredients database

def add_to_ingredients(ingredient_name):
    # Insert the ingredient name into the Ingredients_db collection
    ingredient_document = {
        "name": ingredient_name
    }
    db.Ingredients_db.insert_one(ingredient_document)
    print("Ingredient added to Ingredients_db:", ingredient_document)


# Flask route to move selected ingredients from the shopping list to the user's ingredient list
@app.route('/move_to_ingredients', methods=['POST'])
def move_to_ingredients():
    # Get the selected ingredients from the JSON data sent by the client
    selected_ingredients = request.json
    
    # Remove each selected ingredient from the shopping list session variable
    shopping_list = session.get('shopping_list', [])
    updated_shopping_list = [ingredient for ingredient in shopping_list if ingredient not in selected_ingredients]
    
    # Update the shopping list in the session
    session['shopping_list'] = updated_shopping_list
    
    # Add each selected ingredient to the ingredients database
    for ingredient_name in selected_ingredients:
        add_to_ingredients(ingredient_name)
    
    # Return the updated shopping list along with a success message
    return jsonify({"message": "Ingredients added to ingredients database successfully!", "shopping_list": updated_shopping_list})






 #Flask route to display the ingredients
@app.route('/ingredients')
def display_ingredients():
    # Fetch all user ingredients from MongoDB
    user_ingredients = list(db.Ingredients_db.find({}))
    ingredients_list = [ingredient["name"] for ingredient in user_ingredients]
    return render_template('ingredients.html', ingredients_list=ingredients_list)


    # Flask route to display the shopping list (for debugging)
@app.route('/debug/shopping-list')
def debug_shopping_list():
    shopping_list = session.get('shopping_list', [])
    return jsonify(shopping_list)
# # Flask route to render the shopping list page
# @app.route('/view-shopping-list')
# def view_shopping_list():
#     return render_template('shopping_list.html')

#flask rout to move to my ingredient page from shopping poage 
@app.route('/my-ingredients')
def my_ingredients():
 # Fetch all user ingredients from MongoDB
    user_ingredients = list(db.Ingredients_db.find({}))
    ingredients_list = [ingredient["name"] for ingredient in user_ingredients]
    return render_template('ingredients.html', ingredients_list=ingredients_list)

# Flask route to handle deleting an ingredient from the database
@app.route('/delete-ingredient', methods=['POST'])
def delete_ingredient():
    if request.method == 'POST':
        data = request.json
        ingredient_name = data.get('ingredient_name')
        # Perform deletion operation in the database
        try:
            # Assuming you are using MongoDB, you can use pymongo to delete the ingredient
            result = db.Ingredients_db.delete_one({'name': ingredient_name})
            if result.deleted_count == 1:
                return jsonify({'success': True, 'message': 'Ingredient deleted successfully'})
            else:
                return jsonify({'success': False, 'message': 'Ingredient not found'})
        except Exception as e:
            return jsonify({'success': False, 'message': 'An error occurred: {}'.format(str(e))})

# Flask route to handle adding a recipe to favorites
@app.route('/add-to-favorites', methods=['POST'])
def add_to_favorites_route():
    if request.method == 'POST':
        data = request.json
        recipe_name = data.get('recipe_name')
        # Call the function to add the recipe to favorites
        if add_to_favorites(recipe_name):
            return jsonify({'success': True, 'message': 'Recipe added to favorites'})
        else:
            return jsonify({'success': False, 'message': 'Recipe not found in the database'})

# Function to add a recipe to the user's favorite recipes
def add_to_favorites(recipe_name):
    # Fetch the recipe details from the TestCol collection
    recipe_details = db.TestCol.find_one({"RecipeName": recipe_name})

    # Check if the recipe exists in the TestCol collection
    if recipe_details:
        # Insert the recipe details into the favorite_recipes collection
        db.favorite_recipe.insert_one(recipe_details)
        print("Recipe added to favorite_recipes:", recipe_details)
        return True
    else:
        print("Recipe not found in TestCol:", recipe_name)
        return False


# Flask route to handle removing a recipe from favorites
@app.route('/remove-from-favorites', methods=['POST'])
def remove_from_favorites_route():
    if request.method == 'POST':
        data = request.json
        recipe_name = data.get('recipe_name')
        # Call the function to remove the recipe from favorites
        if remove_from_favorites(recipe_name):
            return jsonify({'success': True, 'message': 'Recipe removed from favorites'})
        else:
            return jsonify({'success': False, 'message': 'Recipe not found in favorites'})

# Function to remove a recipe from favorites
def remove_from_favorites(recipe_name):
    # Delete the recipe from the favorite_recipe collection
    result = db.favorite_recipe.delete_one({"RecipeName": recipe_name})
    if result.deleted_count > 0:
        print("Recipe removed from favorite_recipes:", recipe_name)
        return True
    else:
        print("Recipe not found in favorite_recipes:", recipe_name)
        return False

# Function to calculate total pages for favorite recipes pagination
def calculate_total_pages():
    total_recipes = db.favorite_recipe.count_documents({})
    items_per_page = 10
    total_pages = (total_recipes + items_per_page - 1) // items_per_page
    return total_pages

# Flask route to display favorite recipes
@app.route('/favorite_recipes')
def favorite_recipes():
    favorite_recipes_cursor = db.favorite_recipe.find()
    favorite_recipes = [(recipe["RecipeName"], recipe.get("missing_ingredients_count", 0), recipe.get("image-url", "")) for recipe in favorite_recipes_cursor]

    total_pages = calculate_total_pages()

    def get_recipe_url(recipe_name):
        recipe_details = db.favorite_recipe.find_one({"RecipeName": recipe_name})
        return recipe_details.get("URL", "")

    return render_template('favorite_recipes.html', favorite_recipes=favorite_recipes, total_pages=total_pages, get_recipe_url=get_recipe_url)

# Function to check if a recipe is present in the favorite recipes database
def is_recipe_in_favorites(recipe_name):
    # Check if the recipe exists in the favorite_recipe collection
    recipe_details = db.favorite_recipe.find_one({"RecipeName": recipe_name})
    return recipe_details is not None

# Example usage in a Flask route
@app.route('/check-favorite', methods=['POST'])
def check_favorite():
    if request.method == 'POST':
        data = request.json
        recipe_name = data.get('recipe_name')
        if is_recipe_in_favorites(recipe_name):
            return jsonify({'is_favorite': True})
        else:
            return jsonify({'is_favorite': False})



@app.route('/')
def index():
    return render_template('index.html')

# Flask route to handle displaying the About Us page
@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
    


