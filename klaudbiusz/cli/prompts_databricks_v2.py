"""Databricks v2 prompts - realistic human-style requests for dashboards/apps"""

PROMPTS = {
    # Wanderbricks (Travel Platform)
    "property-search-app": "I need an app to search through properties in the wanderbricks schema. Users should be able to filter by price range, number of bedrooms, and pick a city. Show results on a map with the prices and let them click for details.",

    "booking-calendar": "Build me a booking calendar using data from wanderbricks about reservations or bookings or whatever you call them. Show all bookings for the next few months, color coded by status. When I click a date I should see full booking details and be able to filter by property.",

    "property-comparison": "Build me a tool using the wanderbricks property data where I can select 2-3 properties and compare them side by side - price, amenities, reviews, location. Make it interactive so I can swap properties in and out.",

    "city-performance-app": "Using wanderbricks booking data and property listings, show which cities are making the most money. Let me click into a city to see all properties there, their occupancy rates, and booking trends over time.",

    "guest-booking-history": "Need an app where I can search for any guest in wanderbricks and see their complete booking history - where they stayed, dates, how much they paid, their reviews. Show it as a timeline.",

    "property-pricing-wizard": "Using wanderbricks property info and booking history, build an app that suggests optimal pricing. Show current price vs suggested price based on occupancy and also I want it to have like a slider where I can simulate 'what if I increase price by 10%' and see projected revenue impact. Make it colorful.",

    "host-onboarding-checklist": "Build an app for new hosts using wanderbricks. Show a checklist of things they need to complete - add photos, set pricing, write description. Track completion percentage and show them how similar properties are performing to give them benchmarks.",

    # TPC-DS (Retail)
    "store-manager-simulator": "This is gonna sound weird but using tpcds store sales and stock levels, build an app where I can pretend to be a store manager. Show me my current inventory, sales trends, and let me simulate decisions like 'order more of this product' or 'put this on sale' and show projected outcomes based on historical data.",

    "return-pattern-analysis": "Using tpcds returns data, I need to identify unusual return patterns that might indicate fraud or quality issues. Show customers with high return rates, products that get returned frequently, and any patterns by day of week or time period. Let me set thresholds for what counts as 'high' and flag suspicious cases.",

    "customer-lookalike-finder": "Using tpcds customer data and shopping history, build an app where I input a customer ID and it finds other customers who shop similarly - same categories, similar spend, similar frequency. Show the top 10 matches with a similarity score for targeting similar customer segments.",

    # TPC-H (Supply Chain)
    "parts-catalog-app": "Build a searchable parts catalog from tpch. Users should filter by manufacturer, type, size. Show all suppliers for each part with pricing, and let me compare suppliers.",

    "supplier-risk-assessment": "Using tpch supplier and order data, build a dashboard that scores suppliers by risk. Like if they're always late, or if we depend on them too much, or if they're in a country that's politically unstable (just use the country info as proxy). Color code them red/yellow/green and let me add notes. Also my boss wants a 'export to PDF' button.",

    "part-substitution-finder": "This might be tricky but using tpch parts data, find parts that could substitute for each other - like same size, same type, similar price. Build an app where I search for a part and it suggests alternatives. Include a feature where I can flag two parts as 'definitely substitutable' and save that somehow.",

    "order-archaeology": "Using tpch order history, build something where I can dig into the history of an order - when it was placed, when each item shipped, who touched it, delays, everything. Make it look like a timeline with annotations. Also add a way to compare this order to 'typical' orders to see if it was unusually slow or expensive.",

    # NYC Taxi
    "taxi-zones-map": "Using nyctaxi trip data, build a map showing pickup zones colored by trip volume. Let me filter by time of day and day of week. When I click a zone show average fares and trip counts.",

    "driver-opportunity-map": "Using nyctaxi trip history, build an app showing optimal pickup locations by time of day. Show best pickup zones by hour, average fares, typical trip distances. Let me click on a zone and time to see expected earnings potential and trip frequency.",

    "fare-fairness-checker": "Using nyctaxi trip data, build an app that checks if fares look reasonable for the distance. Like flag trips where the fare seems way too high or too low for the distance traveled. Let me set my own thresholds and show outliers on a map. Also calculate what the 'fair' fare should have been.",

    # Complex / Multi-table
    "property-booking-funnel": "Using wanderbricks user activity data and actual bookings, build a funnel visualization showing the path from property view to booking. Show drop-off rates at each step and let me filter by property type.",

    "host-property-management": "Build a complete host dashboard using wanderbricks. Let hosts see all their properties, upcoming bookings, recent reviews, and revenue metrics in one place.",

    "retail-inventory-sales-reconciliation": "Using tpcds warehouse inventory and sales records, build an app that cross-references inventory movements with actual sales. Flag mismatches and let me drill into specific products and warehouses.",

    "supplier-part-explorer": "Using tpch supplier and parts data, create an interactive app showing relationships between suppliers and parts. Let me search for a part and see all suppliers, or pick a supplier and see all their parts with order history.",

    "omnichannel-customer-view": "Using tpcds customer and sales data from all channels, build an app showing complete customer purchase history across web, store, and catalog. Let me search for customers and see their preferred channel, total spend per channel, and purchase timeline.",

    "review-trends-dashboard": "Using wanderbricks reviews, build a dashboard showing review trends over time. Use review ratings as a sentiment proxy. Show average ratings trending over time, flag properties where ratings are dropping, and let me see the actual review text when I click. Also add a word cloud even though those are kinda useless, my CEO loves them.",

    "cross-platform-customer-analysis": "I want to explore potential overlap between wanderbricks users and tpcds customers. Try matching by email or name to identify customers who both book travel and shop retail. This is experimental since the datasets aren't explicitly linked, but could reveal interesting patterns. Make sure to clearly label this as an experimental analysis.",

    "order-journey-narrative": "Using tpch order and supplier data, build an app that shows the complete journey of an order in a narrative format. Pick an order and present it as a story: 'Customer X ordered parts from 3 different suppliers in 2 countries, Part Y was delayed by 5 days because Supplier Z had issues, total value was $$$'. Make it visual and easy to understand, not just raw data tables.",

    # Simple ML/Predictive
    "price-prediction-tool": "Using wanderbricks property and booking data, build a price prediction tool. Given property characteristics like bedrooms, location, amenities, predict optimal nightly price using linear regression. Show the predicted price and which factors contribute most. Let me adjust property features and see how predicted price changes.",

    "customer-segment-clusters": "Using tpcds customer demographics and purchase behavior, build a customer segmentation tool with K-means clustering. Show 4-5 customer segments with their characteristics - average spend, shopping frequency, preferred categories. Let me explore each cluster and see which customers belong to which segment.",

    "churn-risk-predictor": "Using wanderbricks guest booking history, build a churn prediction tool. Calculate features like days since last booking, total bookings, average spend, and use logistic regression to predict likelihood of booking again. Show customers ranked by churn risk score with their booking patterns.",

    "similar-properties-recommender": "Using wanderbricks property features and booking data, build a recommendation tool using KNN. When I select a property, show the 10 most similar properties based on price, location, amenities, and review scores. Let me adjust which features matter most for similarity.",

    "sales-trend-analyzer": "Using tpcds store sales data, create a trend analysis tool with simple linear regression. For each store or product category, show whether sales are trending up or down and at what rate. Flag categories with declining trends that need attention. Show confidence in the trend prediction.",
}
